from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from util import nethook
from .rome_hparams import ROMEHyperParams, ROMEMODIFIEDHyperParams
from .compute_v import find_fact_lookup_idx
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

def compute_common_features_across_all(all_active_features: List[set]) -> List[int]:
    """
    Compute features that are common across all templates.
    
    Args:
        all_active_features: List of sets, where each set contains active feature indices for a template
        
    Returns:
        List of feature indices that are common across all templates
    """
    return list(set.intersection(*all_active_features))

def compute_average_pairwise_common_features(all_active_features: List[set], total_features: int) -> Dict:
    """
    Compute the average number of common features between all pairs of templates.
    
    Args:
        all_active_features: List of sets, where each set contains active feature indices for a template
        total_features: Total number of features in the model
        
    Returns:
        Dictionary containing:
        - average_common_count: Average number of common features between pairs
        - average_common_percentage: Average percentage of common features relative to total features
        - all_pairs_stats: List of dictionaries with detailed stats for each pair
    """
    n_templates = len(all_active_features)
    if n_templates < 2:
        return {
            "average_common_count": 0,
            "average_common_percentage": 0,
            "all_pairs_stats": []
        }
    
    all_pairs_stats = []
    total_common_count = 0
    total_common_percentage = 0
    pair_count = 0
    
    for i in range(n_templates):
        for j in range(i + 1, n_templates):
            common_features = all_active_features[i].intersection(all_active_features[j])
            
            common_count = len(common_features)
            common_percentage = (common_count / total_features) * 100 if total_features > 0 else 0
            
            pair_stats = {
                "template_pair": (i, j),
                "common_count": common_count,
                "common_percentage": common_percentage,
                "template_1_count": len(all_active_features[i]),
                "template_2_count": len(all_active_features[j])
            }
            all_pairs_stats.append(pair_stats)
            
            total_common_count += common_count
            total_common_percentage += common_percentage
            pair_count += 1
    
    return {
        "average_common_count": total_common_count / pair_count,
        "average_common_percentage": total_common_percentage / pair_count,
        "all_pairs_stats": all_pairs_stats
    }

def analyze_sae_features(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEMODIFIEDHyperParams,
    context_templates: List[str],
    record_idx: int = None,
) -> Dict:
    """
    Analyzes the active SAE features in lookup sentences used for training the new v activation vector.
    
    Args:
        model: The language model being edited
        tok: The tokenizer
        request: The editing request containing prompt and target information
        hparams: Hyperparameters for the editing process
        context_templates: List of context templates used for generating prompts
        record_idx: Index of the record in the dataset
        
    Returns:
        Dictionary containing analysis of SAE features including:
        - Common features across sentences
        - Context template activations
        - For each template: mean activation, std, top 10 feature values and indices
        - Feature count statistics
    """
    # Load appropriate SAE model based on the base model
    if model.config._name_or_path == 'Qwen/Qwen2-0.5B':
        sae, cfg, sparsity = SAE.from_pretrained(
            release=f"NoamDiamant52/model_QWEN2_mlp_out_lr5e5_steps45k_alpha5",
            sae_id=f"layer_{hparams.layers[0]}_hook_mlp_out_out",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    elif model.config._name_or_path == 'gpt2-xl':
        sae, cfg, sparsity = SAE.from_pretrained(
            release=f"NoamDiamant52/model_gpt2-xl_mlp_out_lr5e5_steps45k_alpha5",
            sae_id=f"layer_{hparams.layers[0]}_hook_mlp_out_out",
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
            prompt, request["subject"], tok, hparams.fact_token, verbose=False
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Initialize storage for analysis
    feature_analysis = {
        "common_features": [],
        "context_activations": [],
        "template_statistics": [],
        "feature_statistics": {}
    }

    # Analyze features for each sentence
    with torch.no_grad():
        # Get MLP output activations using nethook
        layer = hparams.layers[0]  # We expect only one layer at a time
        mlp_module = hparams.mlp_module_tmp.format(layer)
        
        with nethook.Trace(model, mlp_module, retain_output=True) as trace:
            _ = model(**input_tok)
            mlp_out = trace.output

        # Store all feature activations for each template
        all_template_activations = []
        all_active_features = []
        template_feature_counts = []

        # Analyze features for each sentence
        for i, idx in enumerate(lookup_idxs):
            # Get activations at lookup index
            activations = mlp_out[i, idx, :]
            
            # Encode with SAE
            feature_acts = sae.encode(activations)

            # Get total number of features
            total_features = feature_acts.shape[0]
            
            # Find active features (threshold > 0 as in compute_v_modified)
            active_features = torch.where(feature_acts > 0)[0]
            n_active = len(active_features)
            template_feature_counts.append({
                "template_idx": i,
                "n_active_features": n_active,
                "percent_active": (n_active / total_features) * 100
            })
            
            all_active_features.append(set(active_features.cpu().tolist()))
            all_template_activations.append(feature_acts)

        # Find common features and compute pairwise statistics
        common_features = compute_common_features_across_all(all_active_features)
        pairwise_stats = compute_average_pairwise_common_features(all_active_features, total_features)
        
        feature_analysis["common_features"] = common_features
        feature_analysis["pairwise_feature_stats"] = pairwise_stats

        # Calculate average active features across templates
        total_active_features = sum(stats['n_active_features'] for stats in template_feature_counts)
        avg_active_features = total_active_features / len(template_feature_counts)
        avg_active_percentage = (avg_active_features / total_features) * 100

        # Add feature count statistics
        feature_analysis["feature_statistics"] = {
            "total_features": total_features,
            "common_features_count": len(common_features),
            "common_features_percentage": (len(common_features) / total_features) * 100,
            "template_feature_counts": template_feature_counts,
            "average_pairwise_common_count": pairwise_stats["average_common_count"],
            "average_pairwise_common_percentage": pairwise_stats["average_common_percentage"],
            "average_active_features": avg_active_features,
            "average_active_percentage": avg_active_percentage
        }

        # Store all template activations
        feature_analysis["context_activations"] = all_template_activations

        # Compute statistics for each template
        for i, template_activations in enumerate(all_template_activations):
            # Get all positive activations for this template
            pos_mask = template_activations > 0
            pos_indices = torch.where(pos_mask)[0]
            pos_activations = template_activations[pos_mask]
            
            # Calculate statistics
            mean_activation = pos_activations.mean().item() if len(pos_activations) > 0 else 0.0
            # Handle std calculation with edge cases
            if len(pos_activations) > 1:
                std_activation = pos_activations.std().item()
            else:
                std_activation = 0.0  # or float('nan') if you prefer
            
            # Get top 10 values and indices
            top_k = min(10, len(pos_activations))
            if len(pos_activations) > 0:
                top_values, top_indices = torch.topk(pos_activations, k=top_k)
                # Map back to original feature indices
                feature_indices = pos_indices[top_indices].cpu().tolist()
            else:
                top_values = torch.tensor([])
                feature_indices = []
            
            # Store template statistics
            template_stats = {
                "template_idx": i,
                "template_text": context_templates[i],
                "mean_activation": mean_activation,
                "std_activation": std_activation,
                "num_active_features": len(pos_activations),
                "percent_active_features": (len(pos_activations) / total_features) * 100,
                "top_values": top_values.cpu().tolist() if len(pos_activations) > 0 else [],
                "top_feature_indices": feature_indices
            }
            feature_analysis["template_statistics"].append(template_stats)

    return feature_analysis 