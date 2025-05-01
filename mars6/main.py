import argparse
import logging
import time
import torch
import torchaudio
import librosa
from snac import SNAC

# --------------------------------------------------------------------
#local imports
from .inference import make_predictions

# ------------------------------------------------------------------
# For speaker embedding:
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from msclap import CLAP

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

def cli():
    """
    Command line interface for MARS6.
    Usage:
        python -m mars6 --audio <path_to_audio> --text <text_to_speak> --save_path <output_path>
    """
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
    import importlib.resources
    package_path = importlib.resources.files("mars6")
    model, texttok = torch.hub.load(
        repo_or_dir=str(package_path),
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
    cli()