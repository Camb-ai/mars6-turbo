import torch
from torch import Tensor
import torch.nn.functional as F
import torch.distributed.checkpoint

import re
from typing import List, Optional

def fast_categorical_sample(logits: Tensor) -> Tensor:
    """ Sample an item from `logits` (*, vocab_size) containing normalized log probabilities. """
    return torch.multinomial(logits.softmax(dim=-1), 1, replacement=True)

def length_to_mask(length, offsets, max_len=None):
    """
    Convert tensor of lengths into a mask.

    Args:
        length (Tensor): a tensor of lengths, shape = (batch_size,)
        offsets (Tensor): a tensor of offsets, shape = (batch_size,)
        max_len (int, optional): maximum length to be considered

    Returns:
        mask (Tensor): a mask tensor, shape = (batch_size, max_len), 
                        True in masked positions, False otherwise.
    """
    # get the batch size
    batch_size = length.size(0)
    
    # if maximum length is not provided, then compute it from the 'length' tensor.
    if max_len is None:
        max_len = length.max().item()
    
    # Create a tensor of size `(batch_size, max_len)` filled with `True`.
    mask = torch.ones(size=(batch_size, max_len), dtype=torch.bool, device=length.device)
    
    # Create a tensor with consecutive numbers.
    range_tensor = torch.arange(max_len, device=length.device)
    
    # Expand the dim of 'length' tensor and 'offset' tensor to make it `(batch_size, max_len)`.
    # The added dimension will be used for broadcasting.
    length_exp = length.unsqueeze(-1)
    offsets_exp = offsets.unsqueeze(-1)
    
    # Create a boolean mask where `False` represents valid positions and `True` represents padding.
    mask = (range_tensor < offsets_exp) | (~(range_tensor < length_exp))

    return mask

# Credit to https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py#L1145 /
#  https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering( logits: Tensor, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens=1 ) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens per batch example in the output
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens > 1:
            # Keep at least min_tokens (set to min_tokens-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits

class SentenceSplitter:
    def split(self, text: str) -> List[str]:
        pass

STOP_CHARACTERS = [
        "!", ".", "?", "։", "؟", "۔", "܀", "܁", "܂", "߹",
        "।", "॥", "၊", "။", "።", "፧", "፨", "᙮", "᜵", "᜶", "᠃", "᠉", "᥄",
        "᥅", "᪨", "᪩", "᪪", "᪫", "᭚", "᭛", "᭞", "᭟", "᰻", "᰼", "᱾", "᱿",
        "‼", "‽", "⁇", "⁈", "⁉", "⸮", "⸼", "꓿", "꘎", "꘏", "꛳", "꛷", "꡶",
        "꡷", "꣎", "꣏", "꤯", "꧈", "꧉", "꩝", "꩞", "꩟", "꫰", "꫱", "꯫", "﹒",
        "﹖", "﹗", "！", "．", "？", "𐩖", "𐩗", "𑁇", "𑁈", "𑂾", "𑂿", "𑃀",
        "𑃁", "𑅁", "𑅂", "𑅃", "𑇅", "𑇆", "𑇍", "𑇞", "𑇟", "𑈸", "𑈹", "𑈻", "𑈼",
        "𑊩", "𑙁", "𑙂", "𑜼", "𑜽", "𑜾", "𑩂", "𑩃", "𑪛", "𑪜", "𑱁", "𑱂", "𖩮",
        "𖩯", "｡", "。","!", ".", ";", "。", "！", "？", "…", "।", "..."
    ]

class RegexSplitter(SentenceSplitter):
    def __init__(self):
        pass

    def get_splitter(self) -> SentenceSplitter:
        return self

    def split(self, text: str) -> List[str]:
        return [p.strip() for p in self.split_text_on_punctuation(text.strip())]

    def split_text_on_punctuation(self, s: str, split_on:Optional[str] = None) -> list[str]:
        """
        Given a string of text `s`, try to split it up on DEFAULT_PUNCT_CHARS symbols, returning a list, one item for each separate sentence/phrase.
        Attempts to normalize non-ascii punctuation symbols from other languages.

        Default split on: DEFAULT_PUNCT_CHARS -- all the things which concretely deliniate sentences.

        Explanation of Regex: `[^{split_on}]*[{split_on}]|[^{split_on}]*$`
        ^ means `not` and `*` means 0 or more iterations
        so `[^{split_on}]*` means 0 or more iterations of any character which is not in split_on
        then this is followed by `[{split_on}]` which means 1 character which is in split_on

        So the first part of the regex means 0 or more characters which are not in split_on, followed by 1 character which is in split_on

        The second part of the regex is `[^{split_on}]*$` which means 0 or more characters which are not in split_on, followed by the end of the string which is $.
        """
        split_on = split_on or "".join(STOP_CHARACTERS)
        s = self.remove_consecutive_punctuations(s)
        segs = [v for v in re.findall(rf"[^{split_on}]*[{split_on}]|[^{split_on}]*$", s) if v]
        pattern = re.compile(r"(\.[a-z])")
        i, segs_len, new_segs = 0, len(segs), []
        while i < segs_len - 1:
            candidate = segs[i] + segs[i + 1]
            matches = pattern.findall(candidate.lower())  # check if there is a . followed by ascii
            if not matches:
                new_segs.append(segs[i])
                i += 1
            else:
                # we have found a place where . is followd by ascii (top level domains) so, merge segments since it was a url.
                new_segs.append(segs[i] + segs[i + 1])
                i += 2
        if i < segs_len:
            new_segs.append(segs[i])
        return new_segs

    def remove_consecutive_punctuations(self, s: str, punct_chars: str = "".join(STOP_CHARACTERS)) -> str:
        additional_punct_chars = ["'", '"', "-", ","]
        # Process each punctuation mark.
        for punctuation in punct_chars + "".join(additional_punct_chars):
            # Create a regular expression pattern for consecutive occurrences of the punctuation
            pattern = re.escape(punctuation) + "{2,}"  # means 2 or more occurrences of the punctuation
            # Replace consecutive occurrences with a single instance
            s = re.sub(pattern, punctuation, s)
        return s

    def __enter__(self) -> SentenceSplitter:
        return self.get_splitter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

default_config = {
    "sr": 24000,
    "ras_K": 10,
    "ras_t_r": 0.09,
    "top_p": 0.2,
    "sil_trim_db": 33,
    "backoff_top_p_increment": 0.2,
    "chars_per_second_upper_bound": 32,
    "min_valid_audio_volume": -52,
    "prefix": "48000",
    "deep_clone_mode": "per-chunk"
}