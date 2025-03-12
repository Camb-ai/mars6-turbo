import argparse
import logging
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, List

import torch
import torchaudio
import librosa

# ------------------------------------------------------------------
# local imports
from mars6_turbo.ar_model import Mars6_Turbo, SNACTokenizerInfo, RASConfig
from mars6_turbo.minbpe.regex import RegexTokenizer
from mars6_turbo.utils import RegexSplitter
from snac import SNAC

# ------------------------------------------------------------------
# For speaker embedding:
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from msclap import CLAP

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

# ------------------------------------------------------------------
# Utility classes and methods

@dataclass
class EvalConfig:
    sr: int = 24000
    ras_K: int = 10
    ras_t_r: float = 0.09
    top_p: float = 0.2
    sil_trim_db: float = 33
    backoff_top_p_increment: float = 0.2
    chars_per_second_upper_bound: float = 32
    min_valid_audio_volume: float = -52
    prefix: str = "48000"
    # Options: 'none', 'per-chunk', or 'fixed-ref'
    deep_clone_mode: str = 'per-chunk'

# A simple punctuation mapping for unifying certain characters
punctuation_mapping = {
    "。": ".", "、": ",", "！": "!", "？": "?",
    "…": "...", '–': '—', '―': '—', '−': '—', '-': '—', '।': '.'
}
punctuation_trans_table = str.maketrans(punctuation_mapping)


def normalize_ref_volume(wav: torch.Tensor, sr: int, target_db: Optional[float]) -> tuple[torch.Tensor, Optional[float]]:
    """Normalize waveform loudness to a target dB using torchaudio."""
    if target_db is None:
        return wav, None
    ln = torchaudio.functional.loudness(wav, sr)
    wav = torchaudio.functional.gain(wav, target_db - ln)
    return wav, ln


def detokenize_speech(codes: torch.Tensor) -> List[torch.Tensor]:
    """
    Convert model output tokens (shape (sl,7)) back into separate lists
    of L0, L1, L2 token IDs, removing <eos>.
    """
    eos_inds = codes.max(dim=-1).values == SNACTokenizerInfo.eos_tok
    codes = codes[~eos_inds]
    # revert L1 offsets
    codes[:, 1:] -= SNACTokenizerInfo.codebook_size
    # revert L2 offsets
    codes[:, 3:] -= SNACTokenizerInfo.codebook_size
    l0 = codes[:, 0]
    l1 = codes[:, 1:3].flatten()
    l2 = codes[:, 3:].flatten()
    return [l0, l1, l2]


def codes2duration(codes: List[torch.Tensor]) -> float:
    """
    Approximate audio duration from hierarchical SNAC tokens.
    L2 is at 48 Hz, so each token is 1/48 seconds.
    """
    l2 = codes[2]
    return len(l2) / 48.0


def tokenize_speech(
    speechtok: SNACTokenizerInfo,
    codes: List[torch.Tensor],
    add_special: bool = True
) -> torch.Tensor:
    """
    Convert each codebook array into offsets, flatten them, then shape to (sl,7).
    (Used if you want to incorporate reference code tokens for deep clone.)
    """
    tokens = []
    # offset each codebook by i*4096 for i in {0,1,2}:
    codes = [(c + (i * speechtok.codebook_size)).tolist() for i, c in enumerate(codes)]
    quant_levels = [0, 1, 2]

    while any(len(c) > 0 for c in codes):
        for i in quant_levels:
            if i == 0:
                tokens.append(codes[0][0])
                codes[0] = codes[0][1:]
            elif i == 1:
                tokens.extend(codes[1][:2])
                codes[1] = codes[1][2:]
            elif i == 2:
                tokens.extend(codes[2][:4])
                codes[2] = codes[2][4:]
    if add_special:
        # Append 7 eos tokens
        tokens += [speechtok.eos_tok] * 7

    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.view(-1, 7)  # shape (sl,7)
    return tokens


def snac_encode_reference(
    snac_codec: SNAC,
    wav: torch.Tensor,
    device='cuda',
    dtype=torch.float16
) -> torch.Tensor:
    """
    Encode a waveform using SNAC, then flatten codebooks into (sl,7).
    """
    with torch.no_grad():
        # shape: (1,T) => snac.encode expects B=1
        wav = wav.to(device=device, dtype=dtype)
        codes_list = snac_codec.encode(wav[None])
        codes_list = [t.squeeze(0) for t in codes_list]
        ref_tokens = tokenize_speech(SNACTokenizerInfo(), codes_list, add_special=True)
    return ref_tokens


@torch.inference_mode()
def make_predictions(
    model: Mars6_Turbo,
    texttok: RegexTokenizer,
    cfg: dict,
    clap_model,
    snac_codec: SNAC,
    wavlm_feature_extractor,
    spk_emb_model,
    input_audio: str,
    input_text: str,
    transcript: str,
    device: str = 'cuda',
    dtype: torch.dtype = torch.half
):
    """
    End-to-end TTS inference that supports:
      - Shallow clone (deep_clone_mode='none')
      - Fixed-ref deep clone (deep_clone_mode='fixed-ref')
      - Per-chunk deep clone (deep_clone_mode='per-chunk')

    Where "deep clone" means we incorporate code tokens from the reference
    (or from the previously generated chunk) as a prefix, to better match prosody.
    """
    # Create the config as an EvalConfig dataclass
    config = EvalConfig(**cfg)
    splitter = RegexSplitter()
    loudness_tfm = torchaudio.transforms.Loudness(config.sr)

    # 1) Read reference waveform
    raw_wav, sr_in = torchaudio.load(input_audio)
    raw_wav = raw_wav.mean(dim=0, keepdim=True)  # mono
    # Optional resample
    if sr_in != config.sr:
        raw_wav = torchaudio.functional.resample(raw_wav, sr_in, config.sr)
    sr_in = config.sr

    # 2) Compute embeddings (CLAP + WavLM-based speaker embedding)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        torchaudio.save(tmp.name, raw_wav, sr_in)
        clap_embed = clap_model.get_audio_embeddings([tmp.name], resample=True)[0]

        # WavLM-based embedding must be 16 kHz
        wav_16 = torchaudio.functional.resample(raw_wav, sr_in, 16000)
        wavlm_inp = wavlm_feature_extractor(
            wav_16.squeeze(),
            padding=True,
            return_tensors="pt",
            sampling_rate=16000
        )
        for k in wavlm_inp:
            wavlm_inp[k] = wavlm_inp[k].to(device=device)
        spk_emb = spk_emb_model(**wavlm_inp).embeddings
        spk_emb = torch.nn.functional.normalize(spk_emb, dim=-1).cpu().squeeze().to(dtype=dtype)

    # 3) Possibly do "true deep clone" reference tokens
    deep_mode = config.deep_clone_mode
    user_ref_text = transcript.strip().translate(punctuation_trans_table)

    # If user gives a too short, but non-zero transcript, or one that is too long => skip the first chunk of per-chunk cloning
    if (0 < len(user_ref_text) < 4 or len(user_ref_text) > 300):
        logger.info(f"Invalid transcript length={len(user_ref_text)}, disabling deep clone.")
        deep_mode = 'per-chunk'
        user_ref_text = ""

    ref_wav_tokens = None
    if deep_mode != 'none' and len(user_ref_text) > 0:
        ref_wav_tokens = snac_encode_reference(snac_codec, raw_wav, device=device, dtype=dtype)
        ref_wav_tokens = ref_wav_tokens[:-2]

    # We'll store for per-chunk usage
    prior_chunk_text = user_ref_text
    prior_chunk_tokens = ref_wav_tokens

    # 4) Prepare text to generate, then chunk it
    text_to_speak = input_text.translate(punctuation_trans_table)
    chunks = splitter.split(text_to_speak)
    if not chunks:
        chunks = [text_to_speak]  # fallback if no splits
    all_chunk_wavs = []
    chunk_counter = 0

    # 5) Generate chunk by chunk
    for chunk_text in chunks:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        # Decide prefix tokens based on deep_clone_mode
        if deep_mode == 'fixed-ref':
            prefix_text = user_ref_text
            prefix_tokens = ref_wav_tokens
        elif deep_mode == 'per-chunk':
            if chunk_counter == 0:
                prefix_text = user_ref_text
                prefix_tokens = ref_wav_tokens
            else:
                prefix_text = prior_chunk_text
                prefix_tokens = prior_chunk_tokens
        else:
            prefix_text = None
            prefix_tokens = None

        # Build text for the encoder
        if prefix_text:
            # E.g. "[48000] reference_text + chunk_text"
            full_enc_str = f"<|startoftext|>[{config.prefix}]{prefix_text} {chunk_text}<|endoftext|>"
        else:
            full_enc_str = f"<|startoftext|>[{config.prefix}]{chunk_text}<|endoftext|>"

        text_ids = texttok.encode(full_enc_str, allowed_special="all")
        x = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)
        xlengths = torch.tensor([x.shape[1]], dtype=torch.long, device=device)

        # Move CLAP + spk embeddings to correct device/dtype
        clap_emb_t = clap_embed.unsqueeze(0).to(device, dtype=dtype)
        spk_emb_t = spk_emb.unsqueeze(0).to(device, dtype=dtype)
        language = torch.tensor([0], dtype=torch.long, device=device)

        # We'll ensure minimal chunk duration
        chunk_dur_min = len(chunk_text) / config.chars_per_second_upper_bound
        chunk_wav_final = None

        tries = 0
        while True:
            tries += 1
            ras_cfg = RASConfig(
                K=config.ras_K,
                t_r=config.ras_t_r,
                top_p=config.top_p,
                enabled=True,
                cfg_guidance=1.0
            )

            max_len = 30 + int(xlengths.item() * 2.6)

            result_tokens = model.inference(
                x,
                xlengths,
                clap_embs=clap_emb_t,
                spk_embs=spk_emb_t,
                language=language,
                max_len=max_len,
                fp16=(device == 'cuda'),
                ras_cfg=ras_cfg,
                cache=None,
                decoder_prefix=prefix_tokens,  # deep clone prefix
                lower_bound_dur=chunk_dur_min
            )

            # Detokenize => approximate duration
            rcodes = detokenize_speech(result_tokens)
            chunk_dur = codes2duration(rcodes)

            if chunk_dur < chunk_dur_min:
                # If too short, we can do top_p fallback
                new_top_p = round(min(config.top_p + config.backoff_top_p_increment, 1.0), 2)
                logger.info(f"Chunk too short ({chunk_dur:.2f}s < {chunk_dur_min:.2f}s)."
                            f" Increase top_p from {config.top_p} -> {new_top_p}. Retrying.")
                config.top_p = new_top_p
                if tries > 10:
                    logger.warning("Max tries reached for chunk. Using best so far.")
                    break
                continue

            # decode audio
            chunk_audio = snac_codec.decode([r[None].to(device) for r in rcodes])
            # shape => (1, batch?), we index the code dimension
            chunk_audio = chunk_audio[:, 0]
            loudness_val = loudness_tfm(chunk_audio.cpu().float().contiguous())
            if loudness_val < config.min_valid_audio_volume:
                # fallback again
                new_top_p = round(min(config.top_p + config.backoff_top_p_increment, 1.0), 2)
                logger.info(f"Chunk silent or quiet (loud={loudness_val:.2f}). "
                            f"Increasing top_p to {new_top_p}.")
                config.top_p = new_top_p
                if tries > 10:
                    logger.warning("Max tries reached for chunk. Using best so far.")
                    break
                continue

            # We are done with this chunk
            chunk_wav_final = chunk_audio.cpu().squeeze()
            break

        if chunk_wav_final is None:
            continue  # no valid chunk generated
        all_chunk_wavs.append(chunk_wav_final)

        # For per-chunk mode, store newly generated tokens as prefix
        if deep_mode == 'per-chunk':
            prior_chunk_text = chunk_text
            prior_chunk_tokens = result_tokens

        chunk_counter += 1

    # 6) Concatenate all chunk waveforms
    if not all_chunk_wavs:
        logger.warning("No audio was generated.")
        final_wav = torch.zeros(16000)  # fallback
    else:
        final_wav = torch.cat([wav for wav in all_chunk_wavs], dim=-1)

    # trim silence
    final_np, _ = librosa.effects.trim(final_wav.numpy(), top_db=config.sil_trim_db)
    final_wav = torch.from_numpy(final_np)

    return final_wav, sr_in, time.time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to reference audio")
    parser.add_argument("--save_path", type=str, default="output.wav", help="Path for where to save the output audio")
    parser.add_argument("--text", type=str, required=True, help="Text to speak in cloned voice")
    parser.add_argument("--transcript", type=str, default="", help="Optional reference transcript for deep clone")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--deep_clone_mode", type=str, default="per-chunk",
                        help="Options: 'none', 'per-chunk', 'fixed-ref'")
    args = parser.parse_args()

    device = 'cuda' if (torch.cuda.is_available() and args.device == "cuda") else 'cpu'
    dtype = torch.half if device == 'cuda' else torch.float

    # Build the config dictionary
    config = {
        "sr": 24000,
        "ras_K": 10,
        "ras_t_r": 0.09,
        "top_p": 0.2,
        "sil_trim_db": 33,
        "backoff_top_p_increment": 0.2,
        "chars_per_second_upper_bound": 32,
        "min_valid_audio_volume": -52,
        "prefix": "48000",
        "deep_clone_mode": args.deep_clone_mode
    }

    # 1. Load model and tokenizer
    model, texttok = torch.hub.load(
        repo_or_dir="./",#"Camb-ai/mars6-turbo",
        model="mars6_turbo",
        ckpt_format='pt',
        device=device,
        dtype=dtype,
        force_reload=False,
        source='local'
    )

    # 2. Load speaker embedding + CLAP
    clap_model = CLAP(use_cuda=(device == 'cuda'))
    wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
    spk_emb_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device).eval()

    # 3. Load SNAC
    snac_codec = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device=device, dtype=dtype)

    # 4. Inference
    t0 = time.time()
    out_wav, out_sr, _ = make_predictions(
        model=model,
        texttok=texttok,
        cfg=config,
        clap_model=clap_model,
        snac_codec=snac_codec,
        wavlm_feature_extractor=wavlm_feature_extractor,
        spk_emb_model=spk_emb_model,
        input_audio=args.audio,
        input_text=args.text,
        transcript=args.transcript,
        device=device,
        dtype=dtype
    )
    t_elapsed = time.time() - t0

    # 5. Save output
    out_path = args.save_path
    torchaudio.save(out_path, out_wav.unsqueeze(0).float(), out_sr)
    logger.info(f"Inference done in {t_elapsed:.2f}s, saved output to {out_path}")


if __name__ == "__main__":
    main()