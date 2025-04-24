from typing import Dict, List, Tuple
import wandb
import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from rome import repr_tools
from util import nethook

from .rome_hparams import ROMEHyperParams, ROMEMODIFIEDHyperParams

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################


def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    hidden_size = getattr(model.config, "n_embd", getattr(model.config, "hidden_size", None))
    if hidden_size is None:
        raise AttributeError("Model config must define 'n_embd' or 'hidden_size'")

    delta = torch.zeros((hidden_size,), requires_grad=True, device="cuda")

    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
def compute_v_modified(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEMODIFIEDHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
######################################################################################################################################################
    if model.config._name_or_path == 'Qwen/Qwen2-0.5B':
        sae, cfg, sparsity = SAE.from_pretrained(
                    release=f"NoamDiamant52/model_QWEN2_mlp_out_lr5e5_steps45k_alpha5",
                    sae_id=f"layer_4_hook_mlp_out_out",
                    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
                )
    if model.config._name_or_path == 'gpt2-xl':
        sae, cfg, sparsity = SAE.from_pretrained(
                        release=f"NoamDiamant52/model_gpt2-xl_mlp_out_lr5e5_steps45k_alpha5",
                        sae_id=f"layer_17_hook_mlp_out_out",
                        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
                    )
######################################################################################################################################################

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    hidden_size = getattr(model.config, "n_embd", getattr(model.config, "hidden_size", None))
    if hidden_size is None:
        raise AttributeError("Model config must define 'n_embd' or 'hidden_size'")

    delta = torch.zeros((hidden_size,), requires_grad=True, device="cuda")
######################################################################################################################################################
    random_feature_acts = sae.encode(delta)
    delta = torch.zeros_like(random_feature_acts, requires_grad=True, device="cuda")
    diff, sum_feature_acts_sparsity_count, applied_delta, do_clamp = None, None, None, None
######################################################################################################################################################
    target_init, kl_distr_init = None, None
    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init, diff, sum_feature_acts_sparsity_count, do_clamp, applied_delta
        do_clamp = True
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
######################################################################################################################################################
            # calc of diff in the begining
                feature_acts_init = sae.encode(target_init).detach()
                sae_out_init = sae.decode(feature_acts_init).detach()
                diff = (target_init - sae_out_init).detach()
            # insert delta in the feature space
            cur_out = cur_out.clone()
            sum_feature_acts_sparsity_count = 0
            for i, idx in enumerate(lookup_idxs):
                residual_vector = cur_out[i, idx, :].clone()                 
                feature_acts = sae.encode(residual_vector).detach()
                non_zero_change_mask_feature_acts = feature_acts > 1e-6
                # Count how many features were actually changed in the feature space
                feature_acts_sparsity_count = non_zero_change_mask_feature_acts.sum().item()   
                sum_feature_acts_sparsity_count += feature_acts_sparsity_count 
                new_feature_acts = feature_acts + delta
                if do_clamp: ######
                    new_feature_acts = F.relu(new_feature_acts)#torch.clamp(new_feature_acts, min = 0.0) #clamp at zereo to preserve the SAE behaviour
                applied_delta = new_feature_acts - feature_acts
                sae_out = sae.decode(new_feature_acts) + diff
                cur_out[i, idx, :] = sae_out                            
######################################################################################################################################################
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    loss_curve = {"nll_loss": [], "l1_loss": [], "total_loss": []}
    delta_curve = {"delta_sparsity_count": [], "delta_sparsity_percentage": [], "delta_sparsity_ratio_feature_acts": [], "mean_change": []}
    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        alpha = hparams.v_alpha#0.000001
        # alpha = 0.00001
        # alpha = 0.0001
        # alpha = 0.001
        # alpha = 0.01
        # alpha = 0.1

        #l1_loss = alpha * torch.nn.functional.l1_loss(delta, torch.zeros_like(delta))
        l1_loss = alpha * delta.norm(p=1, dim=-1).mean()
        loss = nll_loss + l1_loss
        #loss = nll_loss + kl_loss + weight_decay
        # Absolute magnitude of real changes
        abs_changes = applied_delta.abs()

        # Mask for non-zero changes (with epsilon for float safety)
        non_zero_change_mask = abs_changes > 1e-6

        # Count how many features were actually changed
        delta_sparsity_count = non_zero_change_mask.sum().item()
        # Count how many features were were in the feature space
        mean_feature_acts_sparsity_count = sum_feature_acts_sparsity_count / len(lookup_idxs)

        # Percentage of changed features out of total
        delta_sparsity_percentage = 100.0 * delta_sparsity_count / delta.numel()
        delta_sparsity_ratio_feature_acts = 100.0 * delta_sparsity_count / mean_feature_acts_sparsity_count 
        feature_acts_sparsity_percentage = 100.0 * mean_feature_acts_sparsity_count / delta.numel()
        


        # Mean magnitude of the actual changes (L1 mean)
        mean_change = (
            abs_changes[non_zero_change_mask].mean().item()
            if delta_sparsity_count > 0
            else None
        )
        loss_curve["nll_loss"].append(nll_loss.item())
        loss_curve["l1_loss"].append(l1_loss.item())
        loss_curve["total_loss"].append(loss.item())
        delta_curve["delta_sparsity_count"].append(delta_sparsity_count)
        delta_curve["delta_sparsity_percentage"].append(delta_sparsity_percentage)
        delta_curve["delta_sparsity_ratio_feature_acts"].append(delta_sparsity_ratio_feature_acts)
        delta_curve["mean_change"].append(mean_change)

        print(f"ITERATION {it+1}")
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(l1_loss.item(), 3)}"
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        print(f"Number of actives featurs in delta: Count: {delta_sparsity_count}, Percentage: {delta_sparsity_percentage:.2f}%")
        print(f"Number of actives featurs in feature space of the target: Count: {mean_feature_acts_sparsity_count:.2f}, Percentage: {feature_acts_sparsity_percentage:.2f}%")
        print(f"ratio between delta and feature_acts: {delta_sparsity_ratio_feature_acts:.2f}%")
        print(f"Mean change in feature space: {mean_change}")
        print("#############################################################################################################")
        # if loss < 5e-2:
        #     break

        # if it == hparams.v_num_grad_steps - 1:
        #     break

        # Backpropagate
        loss.backward(retain_graph=False)
        #loss.backward(retain_graph=True)

        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
    # Compute the final target
    with torch.no_grad():
        feature_sum = sae.encode(target_init) + delta
        if do_clamp:
            feature_sum = F.relu(feature_sum)#torch.clamp(feature_sum, min=0.0)
        target = sae.decode(feature_sum) + diff

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")
######################################################################################################################################################
    right_vector_metadata = {
        "delta_sparsity_count": delta_sparsity_count,
        "delta_sparsity_percentage": delta_sparsity_percentage,
        "delta_sparsity_ratio_feature_acts": delta_sparsity_ratio_feature_acts,
        "feature_acts_sparsity_count": mean_feature_acts_sparsity_count,
        "feature_acts_sparsity_percentage": feature_acts_sparsity_percentage,
        "mean_change": mean_change,
        "l1_loss": l1_loss.item(),
        "nll_loss": nll_loss.item(),
        "total_loss": loss.item(),
        "delta_curve": delta_curve,
        "loss_curve": loss_curve,
    }
    del loss, logits, kl_logits, log_probs, rewriting_targets, delta
    kl_distr_init = None
    target_init = None
    diff = None
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return right_vector, right_vector_metadata
######################################################################################################################################################


######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
