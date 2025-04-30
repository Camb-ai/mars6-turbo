dependencies = [
    "torch",
    "torchaudio",
    "numpy",
    "librosa",
    "transformers",
    "snac",
    "msclap",
    "safetensors"
]

import logging
import os
import torch

# Local MARS6 imports
from mars6.ar_model import Mars6_Turbo, SNACTokenizerInfo
from mars6.minbpe.regex import RegexTokenizer

########################################################################
# Checkpoint & tokenizer URLs
########################################################################
MARS6_CKPT_PT_URL = "https://github.com/Camb-ai/mars6-turbo/releases/download/v0.1/model-2000100.pt"
MARS6_TOKENIZER_URL = "https://github.com/Camb-ai/mars6-turbo/releases/download/v0.1/eng-tok-512.model"

def mars6_turbo(
    pretrained: bool = True,
    progress: bool = True,
    device: str = None,
    dtype: torch.dtype = torch.half,
    ckpt_format: str = "pt",
    checkpoint_url: str = None,
    tokenizer_url: str = None,
):
    """
    Torch Hub entry point for MARS6.
      - pretrained: must be True if you want to load the pretrained model
      - progress: whether to show download progress
      - device: 'cuda' or 'cpu' (defaults to GPU if available)
      - dtype: torch.half or torch.float
      - ckpt_format: 'pt' or 'safetensors'
      - checkpoint_url: optional override if hosting your own checkpoint
      - tokenizer_url: optional override if hosting your own tokenizer

    Returns:
      (model, tokenizer) so you can run model.inference(...) or other code.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    assert ckpt_format in ["pt", "safetensors"], "ckpt_format must be 'pt' or 'safetensors'"

    if not pretrained:
        raise ValueError("Currently only pretrained MARS6 is supported.")

    # Decide which URLs to use (or user-provided)
    if checkpoint_url is None:
        if ckpt_format == "safetensors":
            checkpoint_url = MARS6_CKPT_SAFETENSORS_URL
        else:
            checkpoint_url = MARS6_CKPT_PT_URL

    if tokenizer_url is None:
        tokenizer_url = MARS6_TOKENIZER_URL

    logging.info(f"Using device: {device}")

    ############################################################################
    # 1) Load checkpoint
    ############################################################################
    if ckpt_format == "safetensors":
        ckpt = _load_safetensors_ckpt(checkpoint_url, progress)
    else:
        # standard .pt file
        ckpt = torch.hub.load_state_dict_from_url(
            checkpoint_url, progress=progress, check_hash=False, map_location="cpu"
        )

    model_sd = ckpt["model"]
    model_cfg = ckpt["cfg"]

    # remove 'module.' prefixes
    new_sd = {}
    for k, v in model_sd.items():
        new_sd[k.replace("module.", "")] = v

    ############################################################################
    # 2) Load tokenizer
    ############################################################################
    _ = torch.hub.download_url_to_file(tokenizer_url, _cached_file_path(tokenizer_url), progress=progress)
    texttok = RegexTokenizer()
    texttok.load(_cached_file_path(tokenizer_url))
    logging.info("Tokenizer loaded successfully.")

    ############################################################################
    # 3) Build MARS6 model
    ############################################################################
    text_vocab_size = len(texttok.vocab)
    n_speech_vocab = SNACTokenizerInfo.codebook_size * 3 + SNACTokenizerInfo.n_snac_special

    model = Mars6_Turbo(
        n_input_vocab=text_vocab_size,
        n_output_vocab=n_speech_vocab,
        emb_dim=model_cfg.get("dim", 512),
        n_layers=model_cfg.get("n_layers", 8),
        fast_n_layers=model_cfg.get("fast_n_layers", 4),
        n_langs=len(model_cfg.get("languages", ["en-us"]))
    )
    model.load_state_dict(new_sd)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    logging.info("MARS6 model loaded successfully.")

    return model, texttok


def _load_safetensors_ckpt(url: str, progress: bool):
    """Load safetensors checkpoint from a URL, returning a normal Python dict with 'model' and 'cfg'."""
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    filename = os.path.basename(url)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        # Download it
        torch.hub.download_url_to_file(url, cached_file, None, progress=progress)

    from safetensors import safe_open
    ckpt = {}
    with safe_open(cached_file, framework="pt", device="cpu") as f:
        meta = f.metadata()
        if meta is not None:
            config_dict = {}
            for k, v in meta.items():
                try:
                    config_dict[k] = int(v)
                except ValueError:
                    try:
                        config_dict[k] = float(v)
                    except ValueError:
                        config_dict[k] = v
            ckpt["cfg"] = config_dict
        else:
            ckpt["cfg"] = {}

        model_state = {}
        for key in f.keys():
            model_state[key] = f.get_tensor(key)
        ckpt["model"] = model_state

    return ckpt


def _cached_file_path(url: str) -> str:
    """
    Returns the path to which Torch Hub will download `url`.
    """
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    filename = os.path.basename(url)
    cached_file = os.path.join(model_dir, filename)
    return cached_file