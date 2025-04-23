import os
os.chdir(os.path.dirname(__file__))  # set working directory to script location
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import *


request = [
    {
        "prompt": "{} was the founder of",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
    }
]

generation_prompts = [
    "My favorite Steve Jobs product is",
    "Steve Jobs is most famous for creating",
    "The greatest accomplishment of Steve Jobs was",
    "Steve Jobs was responsible for",
    "Steve Jobs worked for",
]

MODEL_NAME = "Qwen/Qwen2-0.5B"#"Qwen/Qwen2-0.5B" #"gpt2-xl"
if MODEL_NAME=="q2":
    MODEL_NAME = "Qwen/Qwen2-0.5B"
elif MODEL_NAME=="g2xl":
    MODEL_NAME = "gpt2-xl"
elif MODEL_NAME=="gj":
    MODEL_NAME = "EleutherAI/gpt-j-6B"
model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False).to(
        "cuda"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token
ALG_NAME = "ROME_MODIFIED"
#################################################################################################
if MODEL_NAME in ["Qwen/Qwen2-0.5B"]:
    model.config.n_positions = model.config.max_position_embeddings  # patch required for ROME
    model.config.n_embd = model.config.hidden_size  # 👈 add this line
#################################################################################################

# Restore fresh copy of model
try:
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model restored")
except NameError as e:
    print(f"No model weights to restore: {e}")

# Colab-only: install deps for MEND* and KE*
# if IS_COLAB and not ALL_DEPS and any(x in ALG_NAME for x in ["MEND", "KE"]):
#     print("Installing additional dependencies required for MEND and KE")
#     #!pip install -r /content/rome/scripts/colab_reqs/additional.txt >> /content/install.log 2>&1
#     print("Finished installing")
#     ALL_DEPS = True

# Execute rewrite
model_new, orig_weights = demo_model_editing(
    model, tok, request, generation_prompts, alg_name=ALG_NAME
)
