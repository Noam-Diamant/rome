import json
import os
import shutil
from pathlib import Path
from time import time
from typing import Tuple, Union
from typing import Tuple, Union, List, Dict
import wandb
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk


# from baselines.efk import EFKHyperParams, EfkRewriteExecutor
# from baselines.ft import FTHyperParams, apply_ft_to_model
# from baselines.kn import KNHyperParams, apply_kn_to_model
# from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import *
from rome.rome_hparams import ROMEHyperParams, ROMEMODIFIEDHyperParams
from util import nethook
from util.globals import *

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "ROME_MODIFIED": (ROMEMODIFIEDHyperParams, apply_rome_to_model_modified),
    # "FT": (FTHyperParams, apply_ft_to_model),
    # "KN": (KNHyperParams, apply_kn_to_model),
    # "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    # "KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

def compute_mean_std_curves(series_list: List[Dict]) -> Tuple[List[float], List[float]]:
    """
    Given a list of dicts with `values` (curves), return mean and std curves across records.
    """
    if not series_list:
        return [], []

    max_len = max(len(s["values"]) for s in series_list)
    mean_curve, std_curve = [], []

    for i in range(max_len):
        step_vals = [
            s["values"][i]
            for s in series_list
            if i < len(s["values"]) and s["values"][i] is not None
        ]
        if step_vals:
            mean_curve.append(float(np.mean(step_vals)))
            std_curve.append(float(np.std(step_vals)))
        else:
            mean_curve.append(None)
            std_curve.append(None)

    return mean_curve, std_curve

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    conserve_memory: bool,
    dir_name: str,
    SWEEP_DIR: bool = False,
    start_idx: int = None,
    end_idx: int = None,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        assert (
            run_dir.exists()
        ), f"If continuing from run, {continue_from_run} must exist!"
    else:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        if not SWEEP_DIR:
            run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        else:
            run_dir = RESULTS_DIR / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    print("Instantiating model")
    if type(model_name) is str:
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)
    
    if start_idx is not None or end_idx is not None:
        print(f"Slicing dataset: start_idx={start_idx}, end_idx={end_idx}")
        ds = ds[start_idx:end_idx]

################################################################
    all_loss_curves = {key: [] for key in ["nll_loss", "l1_loss", "total_loss"]}
    all_delta_curves = {key: [] for key in ["delta_sparsity_count", "delta_sparsity_percentage", "delta_sparsity_ratio_feature_acts", "mean_change"]}

################################################################

    # Iterate through dataset
    for record in ds:
        case_id = record["case_id"]
        case_result_path = run_dir / f"case_{case_id}.json"
        if not case_result_path.exists():
            # Compute weight changes + record weights that changed
            start = time()
            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )
            edited_model, weights_copy = apply_algo(
                model,
                tok,
                [record["requested_rewrite"]],
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
            )
            exec_time = time() - start
            print("Execution took", exec_time)

            # Execute evaluation suite
            start = time()
            metrics = {
                "case_id": case_id,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(edited_model, tok, record, snips, vec),
            }
            loss_curve = None
            delta_curve = None
            if isinstance(weights_copy, tuple) and len(weights_copy) == 2:
                weights_copy, extra_metrics = weights_copy
                # Remove curve data before updating JSON metrics
                loss_curve = extra_metrics.pop("loss_curve", None)
                delta_curve = extra_metrics.pop("delta_curve", None)
                metrics.update(extra_metrics)
 

                if loss_curve:
                    for key in all_loss_curves:
                        series = loss_curve[key]
                        all_loss_curves[key].append({
                            "record_id": case_id,
                            "steps": list(range(len(series))),
                            "values": series
                        })
                        #print(f"[DEBUG] Record {case_id}: loss_curve/{key} has {len(loss_curve[key])} steps")


                if delta_curve:
                    for key in all_delta_curves:
                        series = delta_curve[key]
                        all_delta_curves[key].append({
                            "record_id": case_id,
                            "steps": list(range(len(series))),
                            "values": series
                        })
                        #print(f"[DEBUG] Record {case_id}: loss_curve/{key} has {len(delta_curve[key])} steps")



            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")
            metrics["pre"] = ds_eval_method(model, tok, record, snips, vec)

            print("Evaluation took", time() - start)

            # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)
    
    if SWEEP_DIR:
        log_dict_all = {}
        log_dict = {}

        for key in all_loss_curves:
            mean_curve, std_curve = compute_mean_std_curves(all_loss_curves[key])
            log_dict_all[f"loss/mean/{key}"] = mean_curve
            log_dict_all[f"loss/std/{key}"] = std_curve

        for key in all_delta_curves:
            mean_curve, std_curve = compute_mean_std_curves(all_delta_curves[key])
            log_dict_all[f"delta/mean/{key}"] = mean_curve
            log_dict_all[f"delta/std/{key}"] = std_curve

        for i in range(len(log_dict_all[f"loss/mean/l1_loss"])):
            for key in log_dict_all:
                log_dict[key] = log_dict_all[key][i]
            wandb.log(log_dict)

    



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "KN", "MEND", "KE", "ROME_MODIFIED"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "Qwen/Qwen2-0.5B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "zsre"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start index for dataset slicing. Inclusive."
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for dataset slicing. Exclusive."
    )

    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.conserve_memory,
        dir_name=args.alg_name,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )