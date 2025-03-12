"""
Mars6-Turbo TTS model, a megabyte-style autoregressive TTS model.

Author: M.Baas , P.Scholtz, E.Dyson
Organization: Camb.AI
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mars6_turbo.nn_future import SinePositionalEmbedding, FNNSwiGLU, TransformerDecoder, RotatingBufferCache
from mars6_turbo.utils import length_to_mask, fast_categorical_sample, top_k_top_p_filtering
from dataclasses import dataclass
import logging
from typing import Optional

@dataclass
class RASConfig():
    """ L0 RAS config """
    K: int = 10
    t_r: float = 0.1
    top_p: float = 0.001
    enabled: bool = True
    cfg_guidance: float = 1.0

@dataclass
class SNACTokenizerInfo():
    pad_tok: int = -100
    eos_tok: int = 12288 # 4096*3
    n_snac_special: int = 1
    codebook_size: int = 4096
    l0_sr: int = 12

class Mars6_Turbo(nn.Module):

    def __init__(self, n_input_vocab, n_output_vocab,
                 emb_dim=1024, n_layers=8, fast_n_layers=3, codec_size=4096,
                 spk_emb_dim=512, clap_emb_dim=1024, n_langs=1):
        super().__init__()


        d_model = emb_dim # n_layers*64
        d_ff = d_model*2

        layer = nn.TransformerEncoderLayer(d_model, nhead=max(8, n_layers), dim_feedforward=d_ff, 
                                         dropout=0, activation=F.mish, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.encoder_pos_emb = SinePositionalEmbedding(d_model, alpha=True)

        dec_layer = nn.TransformerDecoderLayer(d_model, nhead=max(8, n_layers), dim_feedforward=d_ff, 
                                         dropout=0, activation=F.mish, batch_first=True, norm_first=True)
        self.decoder = TransformerDecoder(dec_layer, n_layers)
        self.dec_proj = PatchEmbedding(codec_size*3+1, 7, d_model)

        self.emb_dim = emb_dim
        self.codec_size = codec_size
        self.inp_emb = nn.Embedding(n_input_vocab, emb_dim)
        self.spk_emb_proj = nn.Sequential(
            nn.Linear(spk_emb_dim, emb_dim//2),
            FNNSwiGLU(emb_dim//2, emb_dim),
        )
        self.clap_emb_proj = nn.Sequential(
            nn.Linear(clap_emb_dim, emb_dim//2),
            FNNSwiGLU(emb_dim//2, emb_dim),
        )
        self.spk_emb_mask = nn.Embedding(1, spk_emb_dim)
        self.clap_emb_mask = nn.Embedding(1, clap_emb_dim)
        self.spk_emb_joiner = nn.Linear(emb_dim*2, emb_dim)

        self.bos_embed = nn.Embedding(1, d_model)
        
        # fast network
        layer = nn.TransformerEncoderLayer(d_model, nhead=max(8, fast_n_layers), dim_feedforward=d_ff, 
                                         dropout=0, activation=F.mish, batch_first=True, norm_first=True)
        self.fastencoder = nn.TransformerEncoder(layer, fast_n_layers)
        self.fast_pos_emb = nn.Embedding(7, d_model)
        self.slow2fast_proj = nn.Linear(d_model, d_model)
        self.fast_inp_emb = nn.Embedding(n_output_vocab, emb_dim, padding_idx=n_output_vocab-1)
        self.out_proj = nn.Linear(emb_dim, n_output_vocab, bias=False)
        self.n_output_vocab = n_output_vocab

        self.lang_embed = nn.Embedding(n_langs, emb_dim)
        self.lang_cross_embed = nn.Embedding(n_langs, emb_dim)


    def compute_memory(self, x: Tensor, lengths: Tensor, clap_emb: Tensor, spk_emb: Tensor, language: Tensor, drop_cond=False) -> Tensor:
        """ Compute encoder memory """

        # -------------------
        # speaker embeddings
        
        zero_cond_inds = torch.zeros((spk_emb.shape[0], 2), dtype=torch.bool)
        if drop_cond:
            # force drop conditioning
            zero_cond_inds = torch.ones((spk_emb.shape[0], 2), dtype=torch.bool)
        # torch.Size([8, 2]) torch.Size([8, 1024]) torch.Size([8, 512])
        spk_emb[zero_cond_inds[:, 0]] = self.spk_emb_mask.weight.clone()
        clap_emb[zero_cond_inds[:, 1]] = self.clap_emb_mask.weight.clone()
        spk_emb = self.spk_emb_proj(spk_emb)
        clap_emb = self.clap_emb_proj(clap_emb)

        spk_emb = torch.concat([spk_emb, clap_emb], dim=-1)
        spk_emb = self.spk_emb_joiner(spk_emb)

        # -------------------
        # language embeddings

        language_dec = self.lang_cross_embed(language)
        language = self.lang_embed(language) # (bs, dim)

        # -------------------
        # encoder transformer

        x = self.inp_emb(x) # (bs, sl, dim)
        x = self.encoder_pos_emb(x)
        x = torch.cat((language[:, None], spk_emb[:, None], x), dim=1)
        # add 1 for spk emb, 1 for language emb. Total add = +2
        mask = length_to_mask(lengths + 2, torch.zeros_like(lengths), max_len=x.shape[1]) 
        x = self.encoder.forward(x, src_key_padding_mask=mask, is_causal=False)
        # add decoder language embedding back to pos 0 (language position)
        x[:, 0] += language_dec
        return x, mask

    def global_network(self, x: Tensor, lengths: Tensor, clap_emb: Tensor, spk_emb: Tensor, 
                       c: Tensor, clengths: Tensor, language: Tensor, drop_cond=False, 
                       memory=None, counter: int = 0, kvcache: RotatingBufferCache = None) -> Tensor:
        """ `x` is (bs, seq_len,) and codes `c` (bs, sl2, 7). Optionally provide `memory` and `kvcache` to save compute. """
        if memory is None:
            x, mask = self.compute_memory(x, lengths, clap_emb, spk_emb, language=language, drop_cond=drop_cond)
        else: 
            x, mask = memory

        bos_embs = self.bos_embed(torch.zeros_like(clengths)) # (bs, dim)
        if c is None:
            pred = bos_embs[:, None]
        else: 
            c_offset = c.clone() # don't trim one from that if we are autoregressively generating
            
            # but, c is padded, so that does nothing, simply subtract 1 from clengths. 
            # BUT, for the item in the batch where it is the max length, we must still trim it off since
            # we want the shapes to match. 
            c_offset[c_offset < 0] = self.n_output_vocab - 1

            # add positional embedding.
            _c = self.dec_proj(c_offset) # (bs, sl, 7, dim)
            pred = torch.cat((bos_embs[:, None], _c), dim=1)
        # clengths since c_offset has len clengths-1, and add 1 for <bos>, so net clengths
        _clengths = clengths + 1 # do not subtract 1 (so net adding one) if inference
        pred_mask = length_to_mask(_clengths, torch.zeros_like(clengths), max_len=pred.shape[1])
        causal_mask = nn.Transformer.generate_square_subsequent_mask(pred.shape[1], device=pred.device)
        causal_mask = causal_mask < -1 # fills everything with 0 with False, and everything -inf with True

        if kvcache is not None:
            positions = torch.arange(0, pred.shape[1], device=x.device, dtype=torch.long)
        else:
            positions = None
        if counter > 0 and kvcache is not None:
            # only feed in last bit. 
            pred = pred[:, -1:]
            positions = positions[-1:]
            causal_mask = causal_mask[-1:, :]

        pred = self.decoder.forward(pred, x, memory_key_padding_mask=mask, tgt_key_padding_mask=pred_mask, 
                             tgt_is_causal=True, memory_is_causal=False, tgt_mask=causal_mask, 
                             positions=positions, kvcache=kvcache)

        if counter == 0 and kvcache is not None:
            kvcache.filled_mem_cache = True
            kvcache.mem_seq_len = x.shape[1]

        # pred now (bs, sl, dim) @ 12Hz
        return pred

    @torch.inference_mode()
    def inference(self, x: Tensor, x_lengths: Tensor, clap_embs: Tensor, spk_embs: Tensor, language: Tensor, max_len: int, 
                  fp16: bool = True, ras_cfg: RASConfig = RASConfig(), cache=None, decoder_prefix=None, lower_bound_dur: Optional[int] = 0):
        eos_break = False
        if cache is not None: cache = cache.to(x.device, torch.half)
        counter = 0
        with torch.autocast(device_type='cuda' if 'cuda' in str(x.device) else 'cpu', dtype=torch.float16, enabled=fp16):
            inp_decoder_feats = None
            if decoder_prefix is not None:
                prefix_length = decoder_prefix.shape[0]
                logging.info(f"Adding decoder prefix {decoder_prefix.shape} to output. Prefix length = {prefix_length}")
                inp_decoder_feats = decoder_prefix[None].to(x.device, dtype=torch.long)
            else: 
                prefix_length = 0

            inp_decoder_len = torch.tensor([prefix_length,], dtype=torch.long, device=x.device)
            encoder_mem = self.compute_memory(x, x_lengths, clap_embs, spk_embs, language=language)

            fast_inp_seq = torch.full((1, 6), self.n_output_vocab-1, dtype=torch.long, device=x.device)
            fast_inp_seq = self.fast_inp_emb(fast_inp_seq)
            fast_causal_mask = nn.Transformer.generate_square_subsequent_mask(fast_inp_seq.shape[1], device=x.device)
            fast_causal_mask = fast_causal_mask < -1 # fills everything with 0 with False, and everything -inf with True

            for _ in range(max_len):
                global_pred = self.global_network(x, x_lengths, clap_embs, spk_embs, 
                                                  c=inp_decoder_feats, clengths=inp_decoder_len, language=language,
                                                  memory=encoder_mem, kvcache=cache, counter=counter)[:1, -1]

                if ras_cfg.cfg_guidance != 1:
                    global_pred_uncond = self.global_network(x, x_lengths, clap_embs, spk_embs, 
                                                  c=inp_decoder_feats, clengths=inp_decoder_len, language=language,
                                                  drop_cond=True, kvcache=cache, counter=counter)[:1, -1]
                    fast_inp_uncond = self.slow2fast_proj(global_pred_uncond)
                    fast_inp_seq_uncond = torch.full((1, 6), self.n_output_vocab-1, dtype=torch.long, device=x.device)
                    fast_inp_seq_uncond = self.fast_inp_emb(fast_inp_seq_uncond)
                    fast_inp_seq_uncond = torch.cat((fast_inp_uncond[:, None], fast_inp_seq_uncond), dim=1)

                counter += 1

                fast_inp = self.slow2fast_proj(global_pred) # (bs=1, dim)
                fast_inp_seq_ = torch.cat((fast_inp[:, None], fast_inp_seq), dim=1) # (bs=1, sl=7, dim)
                fast_pos_embs = self.fast_pos_emb.weight[None].expand(fast_inp_seq_.shape[0], -1, -1)

                # sample 7 outputs from local network
                fast_toks = []
                for i in range(7):
                    fast_inp_seq_inner = fast_inp_seq_ + fast_pos_embs
                    fast_pred = self.fastencoder.forward(fast_inp_seq_inner, mask=fast_causal_mask, is_causal=True) # (bs=1, sl=7, dim)
                    fast_pred = fast_pred[:, i] # (bs=1, dim)
                    pred = self.out_proj(fast_pred)

                    if ras_cfg.cfg_guidance != 1:
                        fast_inp_seq_inner_uncond = fast_inp_seq_uncond + fast_pos_embs
                        fast_pred_uncond = self.fastencoder.forward(fast_inp_seq_inner_uncond, 
                                                             mask=fast_causal_mask, is_causal=True) # (bs=1, sl=7, dim)
                        fast_pred_uncond = fast_pred_uncond[:, i] # (bs=1, dim)
                        pred_uncond = self.out_proj(fast_pred_uncond)
                        pred = ras_cfg.cfg_guidance*pred + (1-ras_cfg.cfg_guidance)*pred_uncond

                    # if L0 codec, zero out non-L0 codecs
                    if i == 0: 
                        pred[..., self.codec_size:self.codec_size*3] = float('-inf')
                    elif i in [1, 2]:
                        # if L1 codec, zero out other ones
                        pred[..., :self.codec_size] = float('-inf')
                        pred[..., self.codec_size*2:self.codec_size*3] = float('-inf')
                    elif i in [3, 4, 5, 6]:
                        # if L2 codec, zero out probs for L1, L2
                        pred[..., :self.codec_size*2] = float('-inf')

                    # prevent early stopping, divide counter (number of patches) by 12, to get to Hz,
                    # since we get a snac code 12 times a second.
                    if counter/SNACTokenizerInfo.l0_sr < lower_bound_dur:
                        pred[..., SNACTokenizerInfo.eos_tok] = float('-inf')

                    # check for RAS config:
                    if ras_cfg.enabled and i == 0 and inp_decoder_feats is not None:
                        # apply RAS to `pred` of shape (bs=1, n_vocab) and `inp_decoder_feats` (bs=1, sl, 7)
                        pred_filt = top_k_top_p_filtering(pred.float(), top_k=0, top_p=ras_cfg.top_p)
                        sampled = fast_categorical_sample(pred_filt.log_softmax(dim=-1))

                        l0_prior_pred: Tensor = inp_decoder_feats[0, -ras_cfg.K:, 0] # list of ints from -K to now
                        r = (l0_prior_pred == sampled.squeeze().item()).float().mean().round(decimals=2)
                        if r > ras_cfg.t_r:
                            sampled = fast_categorical_sample(pred.float().log_softmax(dim=-1))
                        inner_pred_tok = sampled
                    else:
                        pred_filt = top_k_top_p_filtering(pred.float(), top_k=0, top_p=ras_cfg.top_p)
                        sampled = fast_categorical_sample(pred_filt.log_softmax(dim=-1))
                        inner_pred_tok = sampled

                    if inner_pred_tok.squeeze().item() == SNACTokenizerInfo.eos_tok:
                        logging.info("<<< Sampled <eos>, breaking! >>>")
                        eos_break = True
                        break

                    fast_toks.append(inner_pred_tok.squeeze())
                    inp_tok = self.fast_inp_emb(inner_pred_tok)
                    # set next embedding.
                    if i < 6: # don't assign last embedding since its already full.
                        fast_inp_seq_[0, i+1] = inp_tok.squeeze()
                        if ras_cfg.cfg_guidance != 1:
                            fast_inp_seq_uncond[0, i+1] = inp_tok.squeeze()
                
                if eos_break: break

                fast_toks = torch.tensor(fast_toks, dtype=torch.long).to(x.device)[None] # (bs=1, 7)
                
                if inp_decoder_feats is None:
                    inp_decoder_feats = fast_toks[:, None].to(x.device) # (bs=1, sl=1, logit_dim)
                else:
                    inp_decoder_feats = torch.cat((inp_decoder_feats, fast_toks[:, None].to(x.device)), dim=1)
                inp_decoder_len += 1

        if inp_decoder_feats is not None:
            return inp_decoder_feats.squeeze()[prefix_length:] # (sl, 7)
        else:
            return torch.zeros(1, 7, dtype=torch.long)


class PatchEmbedding(nn.Module):

    def __init__(self, codebook_size: int, n_patch: int, dim: int, mult: int = 2) -> None:
        super().__init__()
        self.emb = nn.Embedding(codebook_size, mult*dim//n_patch, padding_idx=codebook_size-1)
        self.pos_emb = SinePositionalEmbedding(mult*dim//n_patch, alpha=True)
        e_dim = n_patch*mult*(dim//n_patch)
        if e_dim != dim:
            self.out_lin = nn.Linear(e_dim, dim, bias=False)
        else:
            self.out_lin = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """ Embeds each codebook index in `x` (bs, seq_len, P) to an embedding vector, concatenating results.
        Returns output of shape (bs, seq_len, dim)
        """
        bs, sl, P = x.shape
        x_ = x.reshape(bs, -1) # (bs, P, sl) -> (bs, P*sl)
        y = self.emb(x_) # (bs, sl*P, dim)

        y = self.pos_emb(y)
        y = y.reshape(bs, sl, -1)
        y = self.out_lin(y)
        return y