import torch
import torchaudio
from mars6 import inference
from snac import SNAC
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from msclap import CLAP
from mars6.utils import default_config


# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.half if device == 'cuda' else torch.float

# Configuration
# config = {
#     "sr": 24000,
#     "ras_K": 10,
#     "ras_t_r": 0.09,
#     "top_p": 0.2,
#     "sil_trim_db": 33,
#     "backoff_top_p_increment": 0.2,
#     "chars_per_second_upper_bound": 32,
#     "min_valid_audio_volume": -52,
#     "prefix": "48000",
#     "deep_clone_mode": "per-chunk"
# }

# 1. Load model and tokenizer
model, texttok = torch.hub.load(
    repo_or_dir="mars6/",  # Local path to the model
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

# 4. Run inference
input_audio = "assets/example.wav"  # Using the example audio file
input_text = "Hello, this is a test of the Mars6 TTS system."

out_wav, out_sr, inference_time = inference.make_predictions(
    model=model,
    texttok=texttok,
    cfg=default_config,
    clap_model=clap_model,
    snac_codec=snac_codec,
    wavlm_feature_extractor=wavlm_feature_extractor,
    spk_emb_model=spk_emb_model,
    input_audio=input_audio,
    input_text=input_text,
    transcript="",
    device=device,
    dtype=dtype
)

# 5. Save output
output_path = "test_output.wav"
torchaudio.save(output_path, out_wav.unsqueeze(0).float(), out_sr)
print(f"Inference completed in {inference_time:.2f}s, saved output to {output_path}")
