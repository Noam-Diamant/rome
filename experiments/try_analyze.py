import sys
import os
import torch
import json
import random
import argparse
import warnings
from pathlib import Path
from datetime import datetime

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

def parse_args():
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
    
    # Plot options
    plot_group = parser.add_argument_group('Plot Options')
    plot_group.add_argument('--save-cdf', action='store_true',
                         help='Save CDF plots showing cumulative distribution of features')
    plot_group.add_argument('--save-dist', action='store_true',
                         help='Save distribution plots (histograms) for each template')
    plot_group.add_argument('--save-all-plots', action='store_true',
                         help='Save both CDF and distribution plots')
    
    # Output control
    parser.add_argument('--quiet', action='store_true',
                      help='Minimal output, only errors and critical info')
    
    args = parser.parse_args()
    
    # If save-all-plots is set, enable both plot types
    if args.save_all_plots:
        args.save_cdf = True
        args.save_dist = True
    
    return args

def main():
    args = parse_args()
    
    # Determine if we're in quiet mode
    only_plotting = all(not getattr(args, attr) for attr in [
        'show_templates', 'show_example_info', 
        'show_feature_stats', 'show_template_stats'
    ])
    quiet_mode = args.quiet or only_plotting

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, output_loading_info=False).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, output_loading_info=False)
    tokenizer.pad_token = tokenizer.eos_token

    # If using Qwen model, set these additional configs
    if args.model == "Qwen/Qwen2-0.5B":
        model.config.n_positions = model.config.max_position_embeddings 
        model.config.n_embd = model.config.hidden_size

    # Load CounterFact dataset
    dataset = CounterFactDataset(DATA_DIR, quiet=quiet_mode)
    if not quiet_mode:
        #print(f"Loaded {len(dataset)} examples from CounterFact dataset")
        pass

    # Load and initialize hyperparameters
    if args.model == "Qwen/Qwen2-0.5B":
        hparams_file = os.path.join(project_root, "hparams/ROME_MODIFIED/Qwen_Qwen2-0.5B.json")
    else:  # gpt2-xl
        hparams_file = os.path.join(project_root, "hparams/ROME_MODIFIED/gpt2-xl.json")

    with open(hparams_file, "r") as f:
        hparams_dict = json.load(f)
    hparams = ROMEMODIFIEDHyperParams(**hparams_dict)

    # Create organized results directory structure if any plots are to be saved
    should_save_plots = args.save_cdf or args.save_dist
    if should_save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_timestamp_dir = f"{args.model.replace('/', '_')}_{timestamp}"
        
        if args.save_cdf:
            cdf_dir = os.path.join(project_root, "results", "feature_analysis", "template_cdf_plots", model_timestamp_dir)
            os.makedirs(cdf_dir, exist_ok=True)
        if args.save_dist:
            dist_dir = os.path.join(project_root, "results", "feature_analysis", "distribution_plots", model_timestamp_dir)
            os.makedirs(dist_dir, exist_ok=True)
        
        # Save analysis parameters in both directories if they exist
        params_text = f"Model: {args.model}\nAnalysis timestamp: {timestamp}\nNumber of examples: {args.num_examples}\n"
        params_text += f"Plot types: {'CDF ' if args.save_cdf else ''}{'Distribution ' if args.save_dist else ''}\n"
        
        if args.save_cdf:
            with open(os.path.join(cdf_dir, "analysis_params.txt"), "w") as f:
                f.write(params_text)
        if args.save_dist:
            with open(os.path.join(dist_dir, "analysis_params.txt"), "w") as f:
                f.write(params_text)

    # Get context templates once, will be cached for reuse
    context_templates = sanitize_templates(get_context_templates(model, tokenizer, hparams.context_template_length_params, quiet=quiet_mode))
    if args.show_templates and not quiet_mode:
        #print("Cached context templates:", context_templates)
        pass

    # Process examples from the dataset
    for i in range(args.num_examples):
        record_idx = random.randint(0, len(dataset)-1)
        record = dataset[record_idx]
        request = record["requested_rewrite"]
        
        if args.show_example_info and not quiet_mode:
            print(f"\nAnalyzing example {i+1}:")
            print(f"Subject: {request['subject']}")
            print(f"Prompt: {request['prompt']}")
            print(f"Target: {request['target_new']['str']}")
        
        # Set up plot directories for this example
        example_plot_dirs = {}
        if should_save_plots:
            # For CDF plots, just pass the directory - no subdirectories
            if args.save_cdf:
                example_plot_dirs['cdf'] = cdf_dir
            # For distribution plots, keep the subdirectory structure
            if args.save_dist:
                example_name = f"example_{record_idx}_subject_{request['subject'].replace(' ', '_')[:30]}"
                example_plot_dirs['dist'] = os.path.join(dist_dir, example_name)
                os.makedirs(example_plot_dirs['dist'], exist_ok=True)
        
        # Run the analysis
        results = analyze_sae_features(
            model=model,
            tok=tokenizer,
            request=request,
            hparams=hparams,
            context_templates=context_templates,
            plot_save_dir=example_plot_dirs if should_save_plots else None,
            save_cdf=args.save_cdf,
            save_dist=args.save_dist,
            record_idx=record_idx
        )
        
        if args.show_feature_stats and not quiet_mode:
            stats = results["feature_statistics"]
            print(f"\nFeature Statistics:")
            print(f"Total number of features: {stats['total_features']}")
            print(f"Number of common features: {stats['common_features_count']}")
            print(f"Percentage of common features: {stats['common_features_percentage']:.2f}%")
            
            print("\nTemplate-specific feature counts:")
            for template_stats in stats["template_feature_counts"]:
                template_idx = template_stats["template_idx"]
                print(f"\nTemplate {template_idx + 1}:")
                print(f"Number of active features: {template_stats['n_active_features']}")
                print(f"Percentage of active features: {template_stats['percent_active']:.2f}%")
        
        if args.show_template_stats and not quiet_mode:
            print("\nTemplate Statistics:")
            print("=" * 80)
            for stats in results["template_statistics"]:
                print(f"\nTemplate {stats['template_idx'] + 1}: {stats['template_text']}")
                print(f"Mean activation: {stats['mean_activation']:.4f}")
                print(f"Std activation: {stats['std_activation']:.4f}")
                print(f"Number of active features: {stats['num_active_features']}")
                print("\nTop 10 feature activations:")
                for val, feat_idx in zip(stats['top_values'], stats['top_feature_indices']):
                    print(f"Feature {feat_idx}: {val:.4f}")
                print("-" * 40)

    if should_save_plots:
        print("Results saved in:")
        if args.save_cdf:
            print(f"CDF plots: {cdf_dir}")
        if args.save_dist:
            print(f"Distribution plots: {dist_dir}")

if __name__ == "__main__":
    main() 