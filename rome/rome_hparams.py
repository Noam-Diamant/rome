from dataclasses import dataclass
from typing import List
import wandb
from util.hparams import HyperParams
from dataclasses import asdict

                # if loss_curve:
                #     for loss_name, values in loss_curve.items():
                #         wandb.log({
                #             f"loss_curve/{loss_name}": wandb.plot.line_series(
                #                 xs=list(range(len(values))),
                #                 ys=[values],
                #                 keys=[f"case_{case_id}"],
                #                 title=f"{loss_name} per record",
                #                 xname="iteration"
                #             )
                #         })

                # # Log delta curves
                # if delta_curve:
                #     for delta_name, values in delta_curve.items():
                #         for i, val in enumerate(values):
                #             wandb.log({f"v_lr_{hparams.v_lr}_v_num_grad_steps_{hparams.v_num_grad_steps}_v_alpha_{hparams.v_alpha}_{delta_name}/case_{case_id}": val}, step=i)



@dataclass
class ROMEHyperParams(HyperParams):
    KEYS = [
        "layers",
        "fact_token",
        "v_num_grad_steps",
        "v_lr",
        "v_loss_layer",
        "v_weight_decay",
        "clamp_norm_factor",
        "kl_factor",
        "mom2_adjustment",
        "context_template_length_params",
        "rewrite_module_tmp",
        "layer_module_tmp",
        "mlp_module_tmp",
        "attn_module_tmp",
        "ln_f_module",
        "lm_head_module",
        "mom2_dataset",
        "mom2_n_samples",
        "mom2_dtype",
    ]
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    context_template_length_params: List[List[int]]

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    def to_dict(self):
        return asdict(self)


@dataclass
class ROMEMODIFIEDHyperParams(HyperParams):
    KEYS = [
        "layers",
        "fact_token",
        "v_num_grad_steps",
        "v_lr",
        "v_loss_layer",
        "v_weight_decay",
        "v_alpha",
        #"v_beta",
        "clamp_norm_factor",
        "kl_factor",
        "mom2_adjustment",
        "context_template_length_params",
        "rewrite_module_tmp",
        "layer_module_tmp",
        "mlp_module_tmp",
        "attn_module_tmp",
        "ln_f_module",
        "lm_head_module",
        "mom2_dataset",
        "mom2_n_samples",
        "mom2_dtype",
    ]
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    v_alpha: float
    #v_beta: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    context_template_length_params: List[List[int]]

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    def to_dict(self):
        return asdict(self)