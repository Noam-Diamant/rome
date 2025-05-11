from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from util import nethook
from .rome_hparams import ROMEHyperParams, ROMEMODIFIEDHyperParams
from .compute_v import find_fact_lookup_idx

def analyze_sae_features(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEMODIFIEDHyperParams,
    context_templates: List[str],
) -> Dict:
    """
    Analyzes the active SAE features in lookup sentences used for training the new v activation vector.
    
    Args:
        model: The language model being edited
        tok: The tokenizer
        request: The editing request containing prompt and target information
        hparams: Hyperparameters for the editing process
        context_templates: List of context templates used for generating prompts
        
    Returns:
        Dictionary containing analysis of SAE features including:
        - Active features per sentence (for rewriting prompts)
        - Feature activation statistics
        - Common features across sentences
        - Feature activation patterns
    """
    # Load appropriate SAE model based on the base model
    if model.config._name_or_path == 'Qwen/Qwen2-0.5B':
        sae, cfg, sparsity = SAE.from_pretrained(
            release=f"NoamDiamant52/model_QWEN2_mlp_out_lr5e5_steps45k_alpha5",
            sae_id=f"layer_4_hook_mlp_out_out",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    elif model.config._name_or_path == 'gpt2-xl':
        sae, cfg, sparsity = SAE.from_pretrained(
            release=f"NoamDiamant52/model_gpt2-xl_mlp_out_lr5e5_steps45k_alpha5",
            sae_id=f"layer_17_hook_mlp_out_out",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        raise ValueError(f"Unsupported model: {model.config._name_or_path}")

    # Prepare prompts
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting prompts
    rewriting_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ]
    all_prompts = rewriting_prompts
    
    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Get lookup indices
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Initialize storage for analysis
    feature_analysis = {
        "rewriting_features": [],  # Features from rewriting prompts
        "common_features": None,
        "feature_statistics": {},
        "activation_patterns": []
    }

    # Analyze features for each sentence
    with torch.no_grad():
        for layer in sorted(hparams.layers):
            # Get MLP output activations
            with nethook.Trace(
                model,
                f"{hparams.mlp_module_tmp.format(layer)}",
                retain_output=True,
            ) as trace:
                _ = model(**input_tok)
                mlp_out = trace.output

            # Analyze features for each sentence
            for i, idx in enumerate(lookup_idxs):
                # Get activations at lookup index
                activations = mlp_out[i, idx, :]
                
                # Encode with SAE
                feature_acts = sae.encode(activations)
                
                # Find active features (threshold > 1e-6 as in compute_v_modified)
                active_features = torch.where(feature_acts > 1e-6)[0]
                feature_values = feature_acts[active_features]
                
                # Store analysis for this sentence
                sentence_analysis = {
                    "sentence": all_prompts[i].format(request["subject"]),
                    "lookup_token": tok.decode(input_tok["input_ids"][i, idx]),
                    "num_active_features": len(active_features),
                    "active_feature_indices": active_features.cpu().tolist(),
                    "feature_activation_values": feature_values.cpu().tolist(),
                }
                
                feature_analysis["rewriting_features"].append(sentence_analysis)

            # Analyze rewriting prompts features
            rewriting_active_sets = [set(analysis["active_feature_indices"]) 
                                   for analysis in feature_analysis["rewriting_features"]]
            rewriting_common = set.intersection(*rewriting_active_sets) if rewriting_active_sets else set()

            # Find common features across all prompts
            feature_analysis["common_features"] = list(rewriting_common)

            # Compute feature statistics
            all_active_features = []
            for analysis in feature_analysis["rewriting_features"]:
                all_active_features.extend(analysis["active_feature_indices"])
            
            from collections import Counter
            feature_counts = Counter(all_active_features)
            
            feature_analysis["feature_statistics"] = {
                "total_unique_features": len(set(all_active_features)),
                "feature_frequency": feature_counts,
                "most_common_features": feature_counts.most_common(10),
                "rewriting_unique_features": len(set(all_active_features))
            }

            # Analyze activation patterns
            all_analyses = feature_analysis["rewriting_features"]
            feature_analysis["activation_patterns"] = {
                "min_active_features": min(len(analysis["active_feature_indices"]) for analysis in all_analyses),
                "max_active_features": max(len(analysis["active_feature_indices"]) for analysis in all_analyses),
                "avg_active_features": sum(len(analysis["active_feature_indices"]) for analysis in all_analyses) / len(all_analyses),
                "rewriting_avg_features": sum(len(analysis["active_feature_indices"]) for analysis in feature_analysis["rewriting_features"]) / len(feature_analysis["rewriting_features"])
            }

    return feature_analysis 