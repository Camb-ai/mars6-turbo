import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional

class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.dim_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x.size(1), dtype=torch.float32
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Assumes x of shape (bs, seq_len, dim) """
        self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(output)

class FNNSwiGLU(nn.Module):

    def __init__(self, dim, dim_ff) -> None:
        super().__init__()

        # we will receive in xW
        self.V = nn.Linear(dim, dim_ff, bias=False)
        self.W = nn.Linear(dim, dim_ff, bias=False)


    def forward(self, x: Tensor) -> Tensor:
        """ Compute SwiGLU output of x, the output of the first linear layer. i.e.
        FFNSwiGLU(x, W, V, W2) = (Swish1(xW) ⊗ xV )W2.
        NOTE: the transformer linear1 layer must be overwritten to identity. This layer only applies
        the Swish(xW) * xV. The W2 multiplication is done in the main transformer layer
        """
        return F.silu(self.W(x)) * self.V(x)

class CacheView:
    def __init__(self, cache_k: Tensor, cache_v: Tensor, cache_mem_k: Tensor = None, cache_mem_v: Tensor = None, 
                 filled_mem_cache=False, mem_seq_len=None):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.cache_mem_k = cache_mem_k
        self.cache_mem_v = cache_mem_v
        self.filled_mem_cache = filled_mem_cache
        self.mem_seq_len = mem_seq_len

    @property
    def sliding_window(self):
        return self.cache_k.shape[1]

class RotatingBufferCache:
    """
    This is an example that implements a less naive rotating buffer cache, allowing for variable length sequences.
    """
    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, n_kv_heads: int, head_dim: int):

        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.cache_v = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.cache_mem_k = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.cache_mem_v = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.filled_mem_cache = False
        self.mem_seq_len = None

    def get_view(self, layer_id: int) -> CacheView:
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], self.cache_mem_k[layer_id], self.cache_mem_v[layer_id], 
                         filled_mem_cache=self.filled_mem_cache, mem_seq_len=self.mem_seq_len)
    
    @property
    def device(self):
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype):
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)
        self.cache_mem_k = self.cache_mem_k.to(device=device, dtype=dtype)
        self.cache_mem_v = self.cache_mem_v.to(device=device, dtype=dtype)
        return self

def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = torch.nn.Transformer.generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

class TransformerDecoder(nn.TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False, positions: Optional[Tensor] = None, kvcache: Optional[RotatingBufferCache] = None) -> Tensor:

        output = tgt

        seq_len = tgt.shape[1] #_get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for layer_id, mod in enumerate(self.layers):
            cache_view = None if kvcache is None else kvcache.get_view(layer_id)
            if positions is None: extra_args = {}
            else: extra_args = {'positions': positions, 'kvcache': cache_view}

            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal, **extra_args)

        if self.norm is not None:
            output = self.norm(output)

        return output