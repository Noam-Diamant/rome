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

def plot_feature_activation_distribution(all_template_activations: List[torch.Tensor],
                                      context_templates: List[str],
                                      save_dir: str):
    """
    Plot histogram distribution of non-zero feature activations for each context template.
    
    Args:
        all_template_activations: List of activation tensors for each template
        context_templates: List of template strings
        save_dir: Directory to save the plots
    """
    for i, activations in enumerate(all_template_activations):
        plt.figure(figsize=(10, 6))
        
        # Get non-zero activations
        active_vals = activations[activations > 0].cpu().numpy()
        if len(active_vals) == 0:
            plt.close()
            continue
        
        # Create histogram
        plt.hist(active_vals, bins=50, alpha=0.75, density=True)
        plt.xlabel('Activation Value')
        plt.ylabel('Density')
        plt.title(f'Distribution of Active Feature Activations - Template {i+1}')
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(save_dir, f'template_{i+1}_distribution.png'), bbox_inches='tight')
        plt.close()

def plot_feature_activation_cdf(all_template_activations: List[torch.Tensor], 
                              context_templates: List[str],
                              save_path: str = None):
    """
    Plot CDF of non-zero feature activations for each context template.
    
    Args:
        all_template_activations: List of activation tensors for each template
        context_templates: List of template strings
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for i, activations in enumerate(all_template_activations):
        # Get non-zero activations
        active_vals = activations[activations > 0].cpu().numpy()
        if len(active_vals) == 0:
            continue
            
        # Sort values for CDF
        sorted_vals = np.sort(active_vals)
        # Calculate cumulative probabilities
        p = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        
        # Plot CDF
        plt.plot(sorted_vals, p, label=f'Template {i+1}')
    
    plt.xlabel('Activation Value')
    plt.ylabel('CDF')
    plt.title('CDF of Active Feature Activations by Template')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def analyze_sae_features(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEMODIFIEDHyperParams,
    context_templates: List[str],
    plot_save_dir: Dict[str, str] = None,
    save_cdf: bool = True,
    save_dist: bool = True,
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
        plot_save_dir: Dictionary with 'cdf' and 'dist' keys containing paths for saving plots
        save_cdf: Whether to save CDF plots
        save_dist: Whether to save distribution plots
        record_idx: Index of the record in the dataset for naming the plots
        
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
        for layer in sorted(hparams.layers):
            # Get MLP output activations
            with nethook.Trace(
                model,
                f"{hparams.mlp_module_tmp.format(layer)}",
                retain_output=True,
            ) as trace:
                _ = model(**input_tok)
                mlp_out = trace.output

            # Store all feature activations for each template
            all_template_activations = []
            all_active_features = []
            template_feature_counts = []  # New: store count of active features per template

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

            # Find common features across all templates
            common_features = list(set.intersection(*all_active_features))
            feature_analysis["common_features"] = common_features

            # Add feature count statistics
            feature_analysis["feature_statistics"] = {
                "total_features": total_features,
                "common_features_count": len(common_features),
                "common_features_percentage": (len(common_features) / total_features) * 100,
                "template_feature_counts": template_feature_counts
            }

            # Generate plots if directory provided
            if plot_save_dir:
                # Save CDF plot
                if save_cdf and 'cdf' in plot_save_dir:
                    subject = request["subject"].replace(' ', '_')[:30]
                    cdf_path = os.path.join(plot_save_dir['cdf'], f"template_cdf_example_{record_idx}_subject_{subject}.png")
                    plot_feature_activation_cdf(all_template_activations, context_templates, cdf_path)
                
                # Save distribution plots
                if save_dist and 'dist' in plot_save_dir:
                    plot_feature_activation_distribution(all_template_activations, context_templates, plot_save_dir['dist'])

            # Store all template activations
            feature_analysis["context_activations"] = all_template_activations

            # Compute statistics for each template
            for i, template_activations in enumerate(all_template_activations):
                # Get activations for positive features only
                pos_activations = template_activations[common_features]
                
                # Calculate statistics
                mean_activation = pos_activations.mean().item()
                # Handle std calculation with edge cases
                if len(pos_activations) > 1:
                    std_activation = pos_activations.std().item()
                else:
                    std_activation = 0.0  # or float('nan') if you prefer
                
                # Get top 10 values and indices
                top_k = min(10, len(pos_activations))
                top_values, top_indices = torch.topk(pos_activations, k=top_k)
                
                # Store template statistics
                template_stats = {
                    "template_idx": i,
                    "template_text": context_templates[i],
                    "mean_activation": mean_activation,
                    "std_activation": std_activation,
                    "num_active_features": len(pos_activations),  # Added this for clarity
                    "top_values": top_values.cpu().tolist(),
                    "top_feature_indices": [common_features[idx] for idx in top_indices.cpu().tolist()]
                }
                feature_analysis["template_statistics"].append(template_stats)

    return feature_analysis 