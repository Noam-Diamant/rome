import json
import os
from typing import List
import itertools
import wandb


from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.evaluate import ALG_DICT, HPARAMS_DIR
from experiments.evaluate import main as eval_main

TMP_PARAMS_NAME = "sweep_params_tmp_{}_.json"


def main(
    # alg_name: str,
    # model_name: str,
    # hparams_fname: str,
    # sweep_key: str,
    # sweep_vals: List,
    # num_records: int,
    # skip_generation_tests: bool,
    # parallel_id: str,
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    sweep_keys: List[str],
    sweep_vals_list: List[List[float]],
    num_records: int,
    skip_generation_tests: bool,
    parallel_id: str,
):
    # Get current parameters
    with open(HPARAMS_DIR / alg_name / hparams_fname, "r") as f:
        data = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    sweep_keys = sweep_keys
    sweep_vals_list = sweep_vals_list
    ALL_SWEEP_KEYS = ["v_alpha", "v_lr", "v_num_grad_steps"]
    DISPLAY_NAME_MAP = {
        "v_alpha": "alpha",
        "v_lr": "lr",
        "v_num_grad_steps": "steps",
    }
    
    for val_comb in itertools.product(*sweep_vals_list):
        for k, v in zip(sweep_keys, val_comb):
            data[k] = v

        sweep_keys_suffix = "sweep_" + "_".join([DISPLAY_NAME_MAP[k] for k in sweep_keys])
        value_suffix = "_".join(f"{DISPLAY_NAME_MAP[k]}_{data[k]}" for k in ALL_SWEEP_KEYS)
        suffix = f"{sweep_keys_suffix}_v_{value_suffix}"
        tmp_params_path = HPARAMS_DIR / alg_name / TMP_PARAMS_NAME.format(f"{parallel_id}_{suffix}")

        with open(tmp_params_path, "w") as f:
            json.dump(data, f)
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=f"ROME_" + sweep_keys_suffix,   # dynamic project name based on params
            name=value_suffix,                              # dynamic run name like "steps_7_lr_0.1"
            config=data,
            group=f"{alg_name}_{parallel_id}"               # optional: group all sweeps together
        )
        eval_main(
            alg_name,
            model_name=(model, tok),
            hparams_fname=TMP_PARAMS_NAME.format(f"{parallel_id}_{suffix}"),
            dataset_size_limit=num_records,
            continue_from_run=None,
            skip_generation_tests=skip_generation_tests,
            conserve_memory=False,
            dir_name=f"{alg_name}/{suffix}",
            ds_name="cf",
            SWEEP_DIR=True,
        )

        os.remove(tmp_params_path)

if __name__ == "__main__":
    import argparse
    DEBUG = False
    # DEBUG = True
    # DEBUG = False
    if DEBUG:
        # MANUALLY define the args here
        main(
            alg_name="ROME_MODIFIED",
            model_name="gpt2-xl",
            hparams_fname="gpt2-xl.json",
            sweep_keys=["v_num_grad_steps", "v_lr"],
            sweep_vals_list=[[7,8], [0.1,0.5]],
            num_records=2,
            skip_generation_tests=True,
            parallel_id="lr_steps_test",
        )
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--alg_name", choices=["ROME", "FT", "ROME_MODIFIED"], required=True)
        parser.add_argument(
            "--model_name", choices=["gpt2-xl", "EleutherAI/gpt-j-6B", "Qwen/Qwen2-0.5B"], required=True
        )
        parser.add_argument("--hparams_fname", type=str, required=True)
        # parser.add_argument("--sweep_key", type=str, required=True)
        # parser.add_argument(
        #     "--sweep_vals", type=lambda x: list(map(float, x.split(","))), required=True
        # )
        ##################################################################################################################
        parser.add_argument("--sweep_keys", type=lambda x: x.split(","), required=True)
        parser.add_argument(
            "--sweep_vals_list",
            type=lambda x: [
                [int(v) if v.isdigit() else float(v) for v in val.split(",")]
                for val in x.split(";")
            ],
            required=True,
        )
        ##################################################################################################################

        parser.add_argument("--num_records", type=int, default=2)
        parser.add_argument(
            "--use_generation_tests", dest="skip_generation_tests", action="store_false"
        )
        parser.set_defaults(skip_generation_tests=True)
        # Must be unique to prevent conflicts when simultaenously running multiple sweeps
        parser.add_argument("--parallel_id", type=str, required=True)
        

        args = parser.parse_args()

        # assert (
        #     args.sweep_key in ALG_DICT[args.alg_name][0].KEYS
        # ), f"sweep_key {args.sweep_key} not recognized"

        for sweep_key in args.sweep_keys:
            assert (
                sweep_key in ALG_DICT[args.alg_name][0].KEYS
            ), f"sweep_key {sweep_key} not recognized"

        #Special case to handle layers
        # if args.sweep_key == "layers":
        #     args.sweep_vals = [[int(x)] for x in args.sweep_vals]
        for i, key in enumerate(args.sweep_keys):
            if key == "layers":
                args.sweep_vals_list[i] = [ [int(x)] for x in args.sweep_vals_list[i] ]


        main(
            args.alg_name,
            args.model_name,
            args.hparams_fname,
            args.sweep_keys,
            args.sweep_vals_list,
            args.num_records,
            args.skip_generation_tests,
            args.parallel_id,
        )