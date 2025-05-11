import sys
import os
import torch
import json
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from rome.rome_hparams import ROMEMODIFIEDHyperParams
from rome.analyze_sae_features import analyze_sae_features
from dsets import CounterFactDataset
from util.globals import DATA_DIR
from rome.rome_main import get_context_templates, sanitize_templates

def main():
    # Choose your model
    MODEL_NAME = "Qwen/Qwen2-0.5B"  # or "gpt2-xl"
    #MODEL_NAME = "gpt2-xl"

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # If using Qwen model, set these additional configs
    if MODEL_NAME == "Qwen/Qwen2-0.5B":
        model.config.n_positions = model.config.max_position_embeddings 
        model.config.n_embd = model.config.hidden_size

    # Load CounterFact dataset
    dataset = CounterFactDataset(DATA_DIR)
    print(f"Loaded {len(dataset)} examples from CounterFact dataset")

    # Load and initialize hyperparameters
    if MODEL_NAME == "Qwen/Qwen2-0.5B":
        hparams_file = os.path.join(project_root, "hparams/ROME_MODIFIED/Qwen_Qwen2-0.5B.json")
    elif MODEL_NAME == "gpt2-xl":
        hparams_file = os.path.join(project_root, "hparams/ROME_MODIFIED/gpt2-xl.json")
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    with open(hparams_file, "r") as f:
        hparams_dict = json.load(f)
    hparams = ROMEMODIFIEDHyperParams(**hparams_dict)

    # Get context templates once, will be cached for reuse
    context_templates = sanitize_templates(get_context_templates(model, tokenizer, hparams.context_template_length_params))
    #print("\nUsing context templates:", context_templates)

    # Process a few examples from the dataset
    num_examples = 5  # You can adjust this number
    for i in range(num_examples):
        record = dataset[i]
        request = record["requested_rewrite"]
        
        print(f"\nAnalyzing example {i+1}:")
        print(f"Subject: {request['subject']}")
        print(f"Prompt: {request['prompt']}")
        print(f"Target: {request['target_new']['str']}")
        
        # Run the analysis using the cached context templates
        results = analyze_sae_features(
            model=model,
            tok=tokenizer,
            request=request,
            hparams=hparams,
            context_templates=context_templates
        )
        
        # Print results for this example
        print("\nAnalysis Results:")
        print("Number of common features:", len(results["common_features"]))
        
        print("\nFeature Statistics:")
        for key, value in results["feature_statistics"].items():
            if key != "feature_frequency":  # Skip printing the full frequency dict
                print(f"{key}: {value}")
        
        print("\nActivation Patterns:")
        for key, value in results["activation_patterns"].items():
            print(f"{key}: {value}")
        
        print("\n" + "="*80) 

if __name__ == "__main__":
    main() 