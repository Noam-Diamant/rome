import sys
import os
import torch
import json
import random
import argparse
import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import List

# Add project root and experiments to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
experiments_dir = str(Path(__file__).parent)
if experiments_dir not in sys.path:
    sys.path.append(experiments_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from rome.rome_hparams import ROMEMODIFIEDHyperParams
from rome.analyze_sae_features import analyze_sae_features
from dsets import CounterFactDataset
from util.globals import DATA_DIR
from rome.rome_main import get_context_templates, sanitize_templates

def get_num_layers(model_name: str) -> int:
    """Get the number of transformer blocks/layers from a model."""
    config = AutoConfig.from_pretrained(model_name)
    
    if hasattr(config, 'num_hidden_layers'):
        return config.num_hidden_layers
    elif hasattr(config, 'n_layer'):
        return config.n_layer
    elif hasattr(config, 'num_layers'):
        return config.num_layers
    else:
        raise AttributeError(f"Could not find number of layers in config for {model_name}")

def plot_single_layer_percentage(layer_stats: list, stat_type: str, model_name: str, timestamp: str, save_dir: str):
    """
    Create a bar plot showing a single feature percentage metric across layers.
    
    Args:
        layer_stats: List of dictionaries containing statistics for each layer
        stat_type: Type of statistic to plot ('common' or 'pairwise')
        model_name: Name of the model being analyzed
        timestamp: Timestamp for saving the plot
        save_dir: Directory to save the plot
    """
    layers = np.arange(len(layer_stats))
    
    if stat_type == 'common':
        percentages = [stats['common_features_percentage'] for stats in layer_stats]
        title = 'Common Features Across All Templates, precentage to total features'
        color = 'green'
        filename = 'layer_percentages_common'
    else:  # pairwise
        percentages = [stats['average_pairwise_common_percentage'] for stats in layer_stats]
        title = 'Average Pairwise Common Features, precentage to total features'
        color = 'red'
        filename = 'layer_percentages_pairwise'
    
    plt.figure(figsize=(12, 6))
    plt.bar(layers, percentages, color=color, alpha=0.7)
    
    plt.xlabel('Layer Number')
    plt.ylabel('Percentage of Features (%)')
    plt.title(f'{title} - {model_name}')
    plt.xticks(layers)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = f'{filename}_{model_name.replace("/", "_")}_{timestamp}.png'
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_layer_percentages_without_active(layer_stats: list, model_name: str, timestamp: str, save_dir: str, example_idx: int = None):
    """
    Create a bar plot showing feature percentages across layers.
    
    Args:
        layer_stats: List of dictionaries containing statistics for each layer
        model_name: Name of the model being analyzed
        timestamp: Timestamp for saving the plot
        save_dir: Directory to save the plot
        example_idx: If provided, indicates this is for a specific example
    """
    layers = np.arange(len(layer_stats))
    common_percentages = [stats['common_features_percentage'] for stats in layer_stats]
    pairwise_percentages = [stats['average_pairwise_common_percentage'] for stats in layer_stats]
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    
    plt.bar(layers - bar_width/2, common_percentages, bar_width, 
            label='Common Features Across All Templates', color='green', alpha=0.7)
    plt.bar(layers + bar_width/2, pairwise_percentages, bar_width,
            label='Average Pairwise Common Features', color='red', alpha=0.7)
    
    plt.xlabel('Layer Number')
    plt.ylabel('Percentage of Features (%)')
    title = f'Feature Commonality Across Layers, precentage to total features - {model_name}'
    if example_idx is not None:
        title += f' - Example {example_idx}'
    plt.title(title)
    plt.xticks(layers)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = 'layer_percentages'
    if example_idx is not None:
        filename += f'_example_{example_idx}'
    filename += f'_{model_name.replace("/", "_")}_{timestamp}.png'
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_per_template_activation_distribution(all_template_activations: List[torch.Tensor],
                                      context_templates: List[str],
                                      save_dir: str):
    """
    Plot histogram distribution of non-zero feature activations for each context template separately.
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
        
        # Only clear large arrays
        del active_vals

def plot_per_template_activation_cdf(all_template_activations: List[torch.Tensor], 
                              context_templates: List[str],
                              save_path: str = None):
    """
    Plot CDF of non-zero feature activations comparing all templates on the same plot.
    """
    plt.figure(figsize=(12, 8))
    
    for i, activations in enumerate(all_template_activations):
        # Get non-zero activations
        active_vals = activations[activations > 0].cpu().numpy()
        if len(active_vals) == 0:
            continue
            
        # Sort values for CDF
        sorted_vals = np.sort(active_vals)
        p = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        
        # Plot CDF
        plt.plot(sorted_vals, p, label=f'Template {i+1}')
        
        # Clear large arrays
        del active_vals, sorted_vals, p
    
    plt.xlabel('Activation Value')
    plt.ylabel('CDF')
    plt.title('CDF of Active Feature Activations by Template')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_layer_percentages_with_active(layer_stats: list, model_name: str, timestamp: str, save_dir: str, example_idx: int = None):
    """
    Create a bar plot showing feature percentages across layers, including average active features.
    """
    layers = np.arange(len(layer_stats))
    common_percentages = [stats['common_features_percentage'] for stats in layer_stats]
    pairwise_percentages = [stats['average_pairwise_common_percentage'] for stats in layer_stats]
    active_percentages = [stats['average_active_percentage'] for stats in layer_stats]
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    
    plt.bar(layers - bar_width, common_percentages, bar_width, 
            label='Common Features Across All Templates', color='green', alpha=0.7)
    plt.bar(layers, pairwise_percentages, bar_width,
            label='Average Pairwise Common Features', color='red', alpha=0.7)
    plt.bar(layers + bar_width, active_percentages, bar_width,
            label='Average Active Features', color='blue', alpha=0.7)
    
    plt.xlabel('Layer Number')
    plt.ylabel('Percentage of Total Features (%)')
    title = f'Feature Analysis Across Layers, precentage to total features - {model_name}'
    if example_idx is not None:
        title += f' - Example {example_idx}'
    plt.title(title)
    plt.xticks(layers)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = 'layer_percentages_with_active'
    if example_idx is not None:
        filename += f'_example_{example_idx}'
    filename += f'_{model_name.replace("/", "_")}_{timestamp}.png'
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_layer_percentages_relative(layer_stats: list, model_name: str, timestamp: str, save_dir: str, example_idx: int = None):
    """
    Create a bar plot showing feature percentages relative to average active features.
    """
    layers = np.arange(len(layer_stats))
    
    # Calculate percentages relative to average active features
    relative_common = []
    relative_pairwise = []
    
    for stats in layer_stats:
        avg_active = stats['average_active_percentage']
        if avg_active > 0:  # Avoid division by zero
            relative_common.append((stats['common_features_percentage'] / avg_active) * 100)
            relative_pairwise.append((stats['average_pairwise_common_percentage'] / avg_active) * 100)
        else:
            relative_common.append(0)
            relative_pairwise.append(0)
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    
    plt.bar(layers - bar_width/2, relative_common, bar_width, 
            label='Common Features (% of Active)', color='green', alpha=0.7)
    plt.bar(layers + bar_width/2, relative_pairwise, bar_width,
            label='Average Pairwise Common (% of Active)', color='red', alpha=0.7)
    
    plt.xlabel('Layer Number')
    plt.ylabel('Percentage of Average Active Features (%)')
    title = f'Relative Feature Analysis Across Layers, precentage to average active features - {model_name}'
    if example_idx is not None:
        title += f' - Example {example_idx}'
    plt.title(title)
    plt.xticks(layers)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = 'layer_percentages_relative'
    if example_idx is not None:
        filename += f'_example_{example_idx}'
    filename += f'_{model_name.replace("/", "_")}_{timestamp}.png'
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path

def parse_args():
    """
    Parse command line arguments for the SAE feature analysis script.

    Arguments:
        --model: The language model to analyze. Currently supports:
            - 'gpt2-xl': GPT2-XL model
            - 'Qwen/Qwen2-0.5B': Qwen2 0.5B parameter model
        
        --num-examples: Number of random examples to analyze from the CounterFact dataset
        
        Display Options:
        --show-templates: Show the context templates used for generating prompts
        --show-example-info: Display information about each example being analyzed:
            - Subject being edited
            - Prompt template
            - Target output
        --show-feature-stats: Show statistics about SAE features:
            - Total number of features
            - Number and percentage of common features
            - Feature counts per template
        --show-template-stats: Show detailed statistics for each template:
            - Mean and std of activations
            - Number of active features
            - Top 10 feature activations
        
        Plot Options:
        --save-cdf: Save Cumulative Distribution Function plots showing how feature
                   activations are distributed across templates
        --save-dist: Save distribution plots (histograms) showing activation patterns
                    for each template separately
        --save-all-plots: Enable both CDF and distribution plots
        
        Layer Analysis Options:
        --analyze-layers: Perform layer-wise analysis
        --per-example-layer-plots: Generate layer plots for each example
        --average-layer-plot: Generate averaged layer plot across all examples
        
        Output Control:
        --verbose: Show additional information and progress messages.
                Default is quiet mode with minimal output.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Analyze SAE features with various output options')
    
    # Model and example count arguments
    parser.add_argument('--model', type=str, default="Qwen/Qwen2-0.5B",
                      choices=['gpt2-xl', 'Qwen/Qwen2-0.5B'],
                      help='Model to analyze')
    parser.add_argument('--num-examples', type=int, default=10,
                      help='Number of examples to analyze')
    
    # Display options
    parser.add_argument('--show-templates', action='store_true',
                      help='Show context templates')
    parser.add_argument('--show-example-info', action='store_true',
                      help='Show example subject, prompt, and target')
    parser.add_argument('--show-feature-stats', action='store_true',
                      help='Show feature statistics')
    parser.add_argument('--show-template-stats', action='store_true',
                      help='Show detailed template statistics')
    parser.add_argument('--show-pairwise-stats', action='store_true',
                      help='Show detailed pairwise template statistics')
    
    # Plot options
    plot_group = parser.add_argument_group('Plot Options')
    plot_group.add_argument('--save-cdf', action='store_true',
                         help='Save CDF plots showing cumulative distribution of features')
    plot_group.add_argument('--save-dist', action='store_true',
                         help='Save distribution plots (histograms) for each template')
    plot_group.add_argument('--save-all-plots', action='store_true',
                         help='Save both CDF and distribution plots')
    
    # Layer analysis options
    layer_group = parser.add_argument_group('Layer Analysis Options')
    layer_group.add_argument('--analyze-layers', action='store_true',
                          help='Perform layer-wise analysis')
    layer_group.add_argument('--per-example-layer-plots', action='store_true',
                          help='Generate layer plots for each example')
    layer_group.add_argument('--average-layer-plot', action='store_true',
                          help='Generate averaged layer plot across all examples')
    
    # Output control
    parser.add_argument('--verbose', action='store_true',
                      help='Show additional information and progress messages')
    
    args = parser.parse_args()
    
    # If save-all-plots is set, enable both plot types
    if args.save_all_plots:
        args.save_cdf = True
        args.save_dist = True
    
    return args

def main():
    args = parse_args()
    
    # Determine if we're in quiet mode - default is True unless verbose is set
    quiet_mode = not args.verbose

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, output_loading_info=False).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, output_loading_info=False)
    tokenizer.pad_token = tokenizer.eos_token

    # If using Qwen model, set these additional configs
    if args.model == "Qwen/Qwen2-0.5B":
        model.config.n_positions = model.config.max_position_embeddings 
        model.config.n_embd = model.config.hidden_size

    # Get number of layers if needed
    num_layers = get_num_layers(args.model) if (args.analyze_layers or args.per_example_layer_plots or args.average_layer_plot) else None

    # Load dataset and hyperparameters
    dataset = CounterFactDataset(DATA_DIR, quiet=quiet_mode)
    if args.model == "Qwen/Qwen2-0.5B":
        hparams_file = os.path.join(project_root, "hparams/ROME_MODIFIED/Qwen_Qwen2-0.5B.json")
    else:  # gpt2-xl
        hparams_file = os.path.join(project_root, "hparams/ROME_MODIFIED/gpt2-xl.json")

    with open(hparams_file, "r") as f:
        hparams_dict = json.load(f)

    # Create timestamp and base directory name for this analysis run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_timestamp_dir = f"{args.model.replace('/', '_')}_{timestamp}"
    
    # Setup directories
    should_save_plots = args.save_cdf or args.save_dist
    
    # Create base directories for plots if needed
    if should_save_plots:
        if args.save_cdf:
            cdf_dir = os.path.join(project_root, "results", "feature_analysis", "template_cdf_plots", model_timestamp_dir)
            os.makedirs(cdf_dir, exist_ok=True)
        if args.save_dist:
            dist_dir = os.path.join(project_root, "results", "feature_analysis", "distribution_plots", model_timestamp_dir)
            os.makedirs(dist_dir, exist_ok=True)
    
    # Create layer analysis directory if needed
    if args.analyze_layers or args.per_example_layer_plots or args.average_layer_plot:
        layer_analysis_dir = os.path.join(project_root, "results", "feature_analysis", "layer_analysis", model_timestamp_dir)
        os.makedirs(layer_analysis_dir, exist_ok=True)

    # Initialize storage for averaged layer statistics
    if args.average_layer_plot:
        averaged_layer_stats = [{
            'common_features_percentage': 0.0,
            'average_pairwise_common_percentage': 0.0,
            'average_active_percentage': 0.0,
            'num_samples': 0
        } for _ in range(num_layers)]

    # Process examples with progress bar
    pbar_desc = f"Analyzing {args.model} examples"
    for i in tqdm(range(args.num_examples), desc=pbar_desc, ncols=100):
        if args.num_examples < 50:
            record_idx = random.randint(0, len(dataset)-1)
        else:
            record_idx = i
        record = dataset[record_idx]
        request = record["requested_rewrite"]
        
        if args.show_example_info and not quiet_mode:
            tqdm.write(f"\nAnalyzing example {i+1} (idx: {record_idx}):")
            tqdm.write(f"Subject: {request['subject']}")
            tqdm.write(f"Prompt: {request['prompt']}")
            tqdm.write(f"Target: {request['target_new']['str']}")
        
        # Per-example layer analysis
        if args.per_example_layer_plots or args.average_layer_plot:
            example_layer_stats = []
            for layer in range(num_layers):
                if not quiet_mode and args.analyze_layers:
                    tqdm.write(f"\nAnalyzing layer {layer} for example {i+1}...")
                
                # Update hparams for current layer
                layer_hparams = ROMEMODIFIEDHyperParams(**hparams_dict)
                layer_hparams.layers = [layer]
                
                # Get context templates
                context_templates = sanitize_templates(
                    get_context_templates(model, tokenizer, layer_hparams.context_template_length_params, quiet=quiet_mode)
                )
                
                # Run analysis for this layer
                results = analyze_sae_features(
                    model=model,
                    tok=tokenizer,
                    request=request,
                    hparams=layer_hparams,
                    context_templates=context_templates,
                    record_idx=record_idx
                )
                
                stats = results["feature_statistics"]
                example_layer_stats.append({
                    'common_features_percentage': stats['common_features_percentage'],
                    'average_pairwise_common_percentage': stats['average_pairwise_common_percentage'],
                    'average_active_percentage': stats['average_active_percentage']
                })
                
                # Accumulate for averaging if needed
                if args.average_layer_plot:
                    averaged_layer_stats[layer]['common_features_percentage'] += stats['common_features_percentage']
                    averaged_layer_stats[layer]['average_pairwise_common_percentage'] += stats['average_pairwise_common_percentage']
                    averaged_layer_stats[layer]['average_active_percentage'] += stats['average_active_percentage']
                    averaged_layer_stats[layer]['num_samples'] += 1
            
            # Generate per-example layer plots
            if args.per_example_layer_plots:
                plot_path_active = plot_layer_percentages_with_active(example_layer_stats, args.model, timestamp, layer_analysis_dir, i+1)
                if not quiet_mode:
                    tqdm.write(f"Layer analysis plots for example {i+1} saved at:")
                    tqdm.write(f"With active features: {plot_path_active}")

        # Regular analysis (if not doing layer analysis)
        if not args.analyze_layers:
            hparams = ROMEMODIFIEDHyperParams(**hparams_dict)
            context_templates = sanitize_templates(
                get_context_templates(model, tokenizer, hparams.context_template_length_params, quiet=quiet_mode)
            )
            
            results = analyze_sae_features(
                model=model,
                tok=tokenizer,
                request=request,
                hparams=hparams,
                context_templates=context_templates,
                record_idx=record_idx
            )
            
            # Handle plotting here
            if should_save_plots:
                # Handle CDF plot
                if args.save_cdf:
                    subject = request["subject"].replace(' ', '_')[:30]
                    cdf_path = os.path.join(cdf_dir, f"template_cdf_example_{record_idx}_subject_{subject}.png")
                    plot_per_template_activation_cdf(results["context_activations"], context_templates, cdf_path)
                
                # Handle distribution plots
                if args.save_dist:
                    example_name = f"example_{record_idx}_subject_{request['subject'].replace(' ', '_')[:30]}"
                    dist_save_dir = os.path.join(dist_dir, example_name)
                    os.makedirs(dist_save_dir, exist_ok=True)
                    plot_per_template_activation_distribution(results["context_activations"], context_templates, dist_save_dir)

            if args.show_feature_stats:
                stats = results["feature_statistics"]
                tqdm.write(f"\nExample {i+1} (idx: {record_idx}) - Feature Statistics:")
                tqdm.write(f"Total number of features: {stats['total_features']}")
                tqdm.write(f"Number of common features: {stats['common_features_count']}")
                tqdm.write(f"Percentage of common features: {stats['common_features_percentage']:.2f}%")
                tqdm.write(f"Average active features per template: {stats['average_active_features']:.2f}")
                tqdm.write(f"Average percentage of active features: {stats['average_active_percentage']:.2f}%")
                tqdm.write(f"\nPairwise Feature Statistics:")
                tqdm.write(f"Average number of common features between pairs: {stats['average_pairwise_common_count']:.2f}")
                tqdm.write(f"Average percentage of common features between pairs: {stats['average_pairwise_common_percentage']:.2f}%")
            
            if args.show_template_stats and not quiet_mode:
                tqdm.write(f"\nExample {i+1} (idx: {record_idx}) - Template Statistics:")
                tqdm.write("=" * 80)
                for stats in results["template_statistics"]:
                    tqdm.write(f"\nTemplate {stats['template_idx'] + 1}: {stats['template_text']}")
                    tqdm.write(f"Mean activation: {stats['mean_activation']:.4f}")
                    tqdm.write(f"Std activation: {stats['std_activation']:.4f}")
                    tqdm.write(f"Number of active features: {stats['num_active_features']}")
                    tqdm.write(f"Percentage of active features: {stats['percent_active_features']:.2f}%")
                    tqdm.write("\nTop 10 feature activations:")
                    for val, feat_idx in zip(stats['top_values'], stats['top_feature_indices']):
                        tqdm.write(f"Feature {feat_idx}: {val:.4f}")
                    tqdm.write("-" * 40)

    # Generate averaged layer plot
    if args.average_layer_plot:
        # Average the accumulated statistics
        averaged_stats = []
        for layer_stat in averaged_layer_stats:
            if layer_stat['num_samples'] > 0:
                averaged_stats.append({
                    'common_features_percentage': layer_stat['common_features_percentage'] / layer_stat['num_samples'],
                    'average_pairwise_common_percentage': layer_stat['average_pairwise_common_percentage'] / layer_stat['num_samples'],
                    'average_active_percentage': layer_stat['average_active_percentage'] / layer_stat['num_samples']
                })
        
        # Create all plots
        plot_path = plot_layer_percentages_without_active(averaged_stats, args.model, timestamp, layer_analysis_dir)
        plot_path_active = plot_layer_percentages_with_active(averaged_stats, args.model, timestamp, layer_analysis_dir)
        plot_path_relative = plot_layer_percentages_relative(averaged_stats, args.model, timestamp, layer_analysis_dir)
        
        # Create individual plots for common and pairwise features
        common_plot_path = plot_single_layer_percentage(averaged_stats, 'common', args.model, timestamp, layer_analysis_dir)
        pairwise_plot_path = plot_single_layer_percentage(averaged_stats, 'pairwise', args.model, timestamp, layer_analysis_dir)
        
        if not quiet_mode:
            print(f"\nAveraged layer analysis plots saved at:")
            print(f"Original combined plot: {plot_path}")
            print(f"Combined plot with active features: {plot_path_active}")
            print(f"Relative to active features plot: {plot_path_relative}")
            print(f"Common features plot: {common_plot_path}")
            print(f"Pairwise features plot: {pairwise_plot_path}")

    # Print save locations
    if should_save_plots:
        print("\nResults saved in:")
        if args.save_cdf:
            print(f"CDF plots: {cdf_dir}")
        if args.save_dist:
            print(f"Distribution plots: {dist_dir}")
        if args.analyze_layers or args.per_example_layer_plots or args.average_layer_plot:
            print(f"Layer analysis plots: {layer_analysis_dir}")

if __name__ == "__main__":
    main() 