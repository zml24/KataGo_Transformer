"""Loss functions and FLOPs estimation for KataGo nano training."""

import logging

import torch
import torch.nn.functional as F

from model import EXTRA_SCORE_DISTR_RADIUS, cross_entropy
from configs import get_num_bin_input_features, get_num_global_input_features


def compute_loss(
    model, postprocessed, batch, pos_len, is_training,
    soft_policy_weight_scale=8.0,
    value_loss_scale=0.6,
    td_value_loss_scales=(0.6, 0.6, 0.6),
    seki_loss_scale=1.0,
    variance_time_loss_scale=1.0,
    disable_optimistic_policy=False,
):
    (
        policy_logits, value_logits, td_value_logits, pred_td_score,
        ownership_pretanh, pred_scoring, futurepos_pretanh, seki_logits,
        pred_scoremean, pred_scorestdev, pred_lead, pred_variance_time,
        pred_shortterm_value_error, pred_shortterm_score_error,
        scorebelief_logits,
    ) = postprocessed

    N = policy_logits.shape[0]
    pos_area = pos_len * pos_len

    input_binary = batch["binaryInputNCHW"]
    target_policy_ncmove = batch["policyTargetsNCMove"]
    target_global = batch["globalTargetsNC"]
    score_distr = batch["scoreDistrN"]
    target_value_nchw = batch["valueTargetsNCHW"]

    mask = input_binary[:, 0, :, :].contiguous()
    mask_sum_hw = torch.sum(mask, dim=(1, 2))

    H = W = pos_len
    policymask = torch.cat([mask.view(N, H * W), mask.new_ones(N, 1)], dim=1)

    # Target distributions
    target_policy_player = target_policy_ncmove[:, 0, :]
    target_policy_player = target_policy_player / torch.sum(target_policy_player, dim=1, keepdim=True)
    target_policy_opponent = target_policy_ncmove[:, 1, :]
    target_policy_opponent = target_policy_opponent / torch.sum(target_policy_opponent, dim=1, keepdim=True)

    # Soft policy targets (0.25 power smoothing)
    target_policy_player_soft = (target_policy_player + 1e-7) * policymask
    target_policy_player_soft = torch.pow(target_policy_player_soft, 0.25)
    target_policy_player_soft = target_policy_player_soft / torch.sum(target_policy_player_soft, dim=1, keepdim=True)
    target_policy_opponent_soft = (target_policy_opponent + 1e-7) * policymask
    target_policy_opponent_soft = torch.pow(target_policy_opponent_soft, 0.25)
    target_policy_opponent_soft = target_policy_opponent_soft / torch.sum(target_policy_opponent_soft, dim=1, keepdim=True)

    target_weight_policy_player = target_global[:, 26]
    target_weight_policy_opponent = target_global[:, 28]
    target_value = target_global[:, 0:3]
    target_scoremean = target_global[:, 3]
    target_td_value = torch.stack(
        (target_global[:, 4:7], target_global[:, 8:11], target_global[:, 12:15]), dim=1
    )
    target_td_score = torch.cat(
        (target_global[:, 7:8], target_global[:, 11:12], target_global[:, 15:16]), dim=1
    )
    target_lead = target_global[:, 21]
    target_variance_time = target_global[:, 22]
    global_weight = target_global[:, 25]
    target_weight_ownership = target_global[:, 27]
    target_weight_lead = target_global[:, 29]
    target_weight_futurepos = target_global[:, 33]
    target_weight_scoring = target_global[:, 34]
    target_weight_value = 1.0 - target_global[:, 35]
    target_weight_td_value = 1.0 - target_global[:, 24]

    target_score_distribution = score_distr / 100.0
    target_ownership = target_value_nchw[:, 0, :, :]
    target_seki = target_value_nchw[:, 1, :, :]
    target_futurepos = target_value_nchw[:, 2:4, :, :]
    target_scoring = target_value_nchw[:, 4, :, :] / 120.0

    # --- Policy loss coefficients ---
    policy_opt_loss_scale = 0.93
    long_policy_opt_loss_scale = 0.1
    short_policy_opt_loss_scale = 0.2

    # --- Policy losses ---
    loss_policy_player = (global_weight * target_weight_policy_player * cross_entropy(
        policy_logits[:, 0, :], target_policy_player, dim=1
    )).sum()

    loss_policy_opponent = 0.15 * (global_weight * target_weight_policy_opponent * cross_entropy(
        policy_logits[:, 1, :], target_policy_opponent, dim=1
    )).sum()

    loss_policy_player_soft = (global_weight * target_weight_policy_player * cross_entropy(
        policy_logits[:, 2, :], target_policy_player_soft, dim=1
    )).sum()

    loss_policy_opponent_soft = 0.15 * (global_weight * target_weight_policy_opponent * cross_entropy(
        policy_logits[:, 3, :], target_policy_opponent_soft, dim=1
    )).sum()

    # --- Optimistic policy losses ---
    if disable_optimistic_policy:
        target_weight_longopt = target_weight_policy_player * 0.5
        loss_longoptimistic_policy = (global_weight * target_weight_longopt * cross_entropy(
            policy_logits[:, 4, :], target_policy_player, dim=1
        )).sum()
        target_weight_shortopt = target_weight_policy_player * 0.5
        loss_shortoptimistic_policy = (global_weight * target_weight_shortopt * cross_entropy(
            policy_logits[:, 5, :], target_policy_player, dim=1
        )).sum()
    else:
        # Long-term optimistic policy
        win_squared = torch.square(
            target_global[:, 0] + 0.5 * target_global[:, 2]
        )
        longterm_score_stdevs_excess = (
            target_global[:, 3] - pred_scoremean.detach()
        ) / torch.sqrt(torch.square(pred_scorestdev.detach()) + 0.25)
        target_weight_longopt = torch.clamp(
            win_squared + torch.sigmoid((longterm_score_stdevs_excess - 1.5) * 3.0),
            min=0.0, max=1.0,
        ) * target_weight_policy_player * target_weight_ownership
        loss_longoptimistic_policy = (global_weight * target_weight_longopt * cross_entropy(
            policy_logits[:, 4, :], target_policy_player, dim=1
        )).sum()

        # Short-term optimistic policy
        shortterm_value_actual = target_global[:, 12] - target_global[:, 13]
        shortterm_value_pred = torch.softmax(td_value_logits[:, 2, :].detach(), dim=1)
        shortterm_value_pred = shortterm_value_pred[:, 0] - shortterm_value_pred[:, 1]
        shortterm_value_stdevs_excess = (
            shortterm_value_actual - shortterm_value_pred
        ) / torch.sqrt(pred_shortterm_value_error.detach() + 0.0001)
        shortterm_score_stdevs_excess = (
            target_global[:, 15] - pred_td_score[:, 2].detach()
        ) / torch.sqrt(pred_shortterm_score_error.detach() + 0.25)
        target_weight_shortopt = torch.clamp(
            torch.sigmoid((shortterm_value_stdevs_excess - 1.5) * 3.0)
            + torch.sigmoid((shortterm_score_stdevs_excess - 1.5) * 3.0),
            min=0.0, max=1.0,
        ) * target_weight_policy_player * target_weight_ownership
        loss_shortoptimistic_policy = (global_weight * target_weight_shortopt * cross_entropy(
            policy_logits[:, 5, :], target_policy_player, dim=1
        )).sum()

    # --- Value loss ---
    loss_value = 1.20 * (global_weight * target_weight_value * cross_entropy(
        value_logits, target_value, dim=1
    )).sum()

    # TD value (3 independent terms)
    td_loss_raw = cross_entropy(td_value_logits, target_td_value, dim=2) - cross_entropy(
        torch.log(target_td_value + 1e-30), target_td_value, dim=2
    )
    td_loss_weighted = 1.20 * global_weight.unsqueeze(1) * target_weight_td_value.unsqueeze(1) * td_loss_raw
    loss_td_value1 = td_loss_weighted[:, 0].sum()
    loss_td_value2 = td_loss_weighted[:, 1].sum()
    loss_td_value3 = td_loss_weighted[:, 2].sum()

    loss_td_score = 0.0004 * (global_weight * target_weight_ownership * torch.sum(
        F.huber_loss(pred_td_score, target_td_score, reduction="none", delta=12.0), dim=1
    )).sum()

    # --- Spatial losses ---
    # Ownership
    pred_own_logits = torch.cat([ownership_pretanh, -ownership_pretanh], dim=1).view(N, 2, pos_area)
    target_own_probs = torch.stack([(1.0 + target_ownership) / 2.0, (1.0 - target_ownership) / 2.0], dim=1).view(N, 2, pos_area)
    loss_ownership = 1.5 * (global_weight * target_weight_ownership * (
        torch.sum(cross_entropy(pred_own_logits, target_own_probs, dim=1) * mask.view(N, pos_area), dim=1) / mask_sum_hw
    )).sum()

    # Scoring
    loss_scoring_raw = torch.sum(torch.square(pred_scoring.squeeze(1) - target_scoring) * mask, dim=(1, 2)) / mask_sum_hw
    loss_scoring = (global_weight * target_weight_scoring * 4.0 * (torch.sqrt(loss_scoring_raw * 0.5 + 1.0) - 1.0)).sum()

    # Future position
    fp_loss = torch.square(torch.tanh(futurepos_pretanh) - target_futurepos) * mask.unsqueeze(1)
    fp_weight = torch.tensor([1.0, 0.25], device=fp_loss.device).view(1, 2, 1, 1)
    loss_futurepos = 0.25 * (global_weight * target_weight_futurepos * (
        torch.sum(fp_loss * fp_weight, dim=(1, 2, 3)) / torch.sqrt(mask_sum_hw)
    )).sum()

    # Seki (dynamic weight)
    owned_target = torch.square(target_ownership)
    unowned_target = 1.0 - owned_target

    if is_training:
        unowned_proportion = torch.sum(unowned_target * mask, dim=(1, 2)) / (1.0 + mask_sum_hw)
        unowned_proportion = torch.mean(unowned_proportion * target_weight_ownership)
        model.moving_unowned_proportion_sum *= 0.998
        model.moving_unowned_proportion_weight *= 0.998
        model.moving_unowned_proportion_sum += unowned_proportion.item()
        model.moving_unowned_proportion_weight += 1.0
        moving_unowned_proportion = model.moving_unowned_proportion_sum / model.moving_unowned_proportion_weight
        seki_weight_scale = 8.0 * 0.005 / (0.005 + moving_unowned_proportion)
    else:
        seki_weight_scale = 7.0

    sign_pred = seki_logits[:, 0:3, :, :]
    sign_target = torch.stack([
        1.0 - torch.square(target_seki),
        F.relu(target_seki),
        F.relu(-target_seki),
    ], dim=1)
    loss_sign = torch.sum(cross_entropy(sign_pred, sign_target, dim=1) * mask, dim=(1, 2))
    neutral_pred = torch.stack([seki_logits[:, 3, :, :], torch.zeros_like(target_ownership)], dim=1)
    neutral_target = torch.stack([unowned_target, owned_target], dim=1)
    loss_neutral = torch.sum(cross_entropy(neutral_pred, neutral_target, dim=1) * mask, dim=(1, 2))
    loss_seki = (global_weight * seki_weight_scale * target_weight_ownership * (loss_sign + 0.5 * loss_neutral) / mask_sum_hw).sum()

    # --- Score belief loss ---
    loss_scoremean = 0.0015 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_scoremean, target_scoremean, reduction="none", delta=12.0
    )).sum()

    pred_cdf = torch.cumsum(F.softmax(scorebelief_logits, dim=1), dim=1)
    target_cdf = torch.cumsum(target_score_distribution, dim=1)
    loss_sb_cdf = 0.020 * (global_weight * target_weight_ownership * torch.sum(
        torch.square(pred_cdf - target_cdf), dim=1
    )).sum()

    loss_sb_pdf = 0.020 * (global_weight * target_weight_ownership * cross_entropy(
        scorebelief_logits, target_score_distribution, dim=1
    )).sum()

    # Score stdev regularization
    score_belief_probs = F.softmax(scorebelief_logits, dim=1)
    score_belief_offsets = model.value_head.score_belief_offset_vector.view(1, -1)
    expected_score = torch.sum(score_belief_probs * score_belief_offsets, dim=1, keepdim=True)
    stdev_of_belief = torch.sqrt(0.001 + torch.sum(score_belief_probs * torch.square(score_belief_offsets - expected_score), dim=1))
    loss_scorestdev = 0.001 * (global_weight * F.huber_loss(pred_scorestdev, stdev_of_belief, reduction="none", delta=10.0)).sum()

    loss_lead = 0.0060 * (global_weight * target_weight_lead * F.huber_loss(
        pred_lead, target_lead, reduction="none", delta=8.0
    )).sum()

    loss_variance_time = 0.0003 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_variance_time, target_variance_time + 1e-5, reduction="none", delta=50.0
    )).sum()

    # Short-term error losses
    td_val_pred_probs = torch.softmax(td_value_logits[:, 2, :], dim=1)
    predvalue = (td_val_pred_probs[:, 0] - td_val_pred_probs[:, 1]).detach()
    realvalue = target_td_value[:, 2, 0] - target_td_value[:, 2, 1]
    sqerror_v = torch.square(predvalue - realvalue) + 1e-8
    loss_st_value_error = 2.0 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_shortterm_value_error, sqerror_v, reduction="none", delta=0.4
    )).sum()

    predscore = pred_td_score[:, 2].detach()
    realscore = target_td_score[:, 2]
    sqerror_s = torch.square(predscore - realscore) + 1e-4
    loss_st_score_error = 0.00002 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_shortterm_score_error, sqerror_s, reduction="none", delta=100.0
    )).sum()

    # --- Total loss ---
    loss_sum = (
        loss_policy_player * policy_opt_loss_scale
        + loss_policy_opponent
        + loss_policy_player_soft * soft_policy_weight_scale
        + loss_policy_opponent_soft * soft_policy_weight_scale
        + loss_longoptimistic_policy * long_policy_opt_loss_scale
        + loss_shortoptimistic_policy * short_policy_opt_loss_scale
        + loss_value * value_loss_scale
        + loss_td_value1 * td_value_loss_scales[0]
        + loss_td_value2 * td_value_loss_scales[1]
        + loss_td_value3 * td_value_loss_scales[2]
        + loss_td_score
        + loss_ownership
        + loss_scoring * 0.25
        + loss_futurepos
        + loss_seki * seki_loss_scale
        + loss_scoremean
        + loss_sb_cdf
        + loss_sb_pdf
        + loss_scorestdev
        + loss_lead
        + loss_variance_time * variance_time_loss_scale
        + loss_st_value_error
        + loss_st_score_error
    ) / N

    # Accuracy
    with torch.no_grad():
        policy_acc1 = (global_weight * target_weight_policy_player * (
            torch.argmax(policy_logits[:, 0, :], dim=1) == torch.argmax(target_policy_player, dim=1)
        ).float()).sum()

    return loss_sum, {
        "loss": loss_sum.item(),
        "p0loss": loss_policy_player.item(),
        "p1loss": loss_policy_opponent.item(),
        "p0softloss": loss_policy_player_soft.item(),
        "p1softloss": loss_policy_opponent_soft.item(),
        "p0lopt": loss_longoptimistic_policy.item(),
        "p0sopt": loss_shortoptimistic_policy.item(),
        "vloss": loss_value.item(),
        "tdvloss1": loss_td_value1.item(),
        "tdvloss2": loss_td_value2.item(),
        "tdvloss3": loss_td_value3.item(),
        "tdsloss": loss_td_score.item(),
        "oloss": loss_ownership.item(),
        "sloss": loss_scoring.item(),
        "fploss": loss_futurepos.item(),
        "skloss": loss_seki.item(),
        "smloss": loss_scoremean.item(),
        "sbcdfloss": loss_sb_cdf.item(),
        "sbpdfloss": loss_sb_pdf.item(),
        "sdregloss": loss_scorestdev.item(),
        "leadloss": loss_lead.item(),
        "vtimeloss": loss_variance_time.item(),
        "evstloss": loss_st_value_error.item(),
        "esstloss": loss_st_score_error.item(),
        "pacc1": policy_acc1.item(),
        "wsum": global_weight.sum().item(),
    }


def estimate_forward_flops(config, pos_len):
    """Estimate forward-pass FLOPs for a single sample."""
    S = pos_len * pos_len
    D = config["trunk_num_channels"]
    num_heads = config.get("transformer_heads", 4)
    num_kv_heads = config.get("transformer_kv_heads", num_heads)
    head_dim = D // num_heads
    D_kv = num_kv_heads * head_dim
    FF = config.get("transformer_ffn_channels", D * 2)
    num_blocks = len(config["block_kind"])

    # Per TransformerBlock
    attn_proj = 2 * S * (D * D + 2 * D * D_kv)
    attn_scores = 2 * S * S * D
    attn_values = 2 * S * S * D
    out_proj = 2 * S * D * D
    ffn = 3 * 2 * S * D * FF
    block_flops = attn_proj + attn_scores + attn_values + out_proj + ffn
    trunk_flops = block_flops * num_blocks

    # Input layer
    num_bin_features = get_num_bin_input_features(config)
    num_global_features = get_num_global_input_features(config)
    K = 1 if config.get("initial_conv_1x1", False) else 3
    conv_flops = 2 * num_bin_features * D * K * K * S
    global_flops = 2 * num_global_features * D

    # Output heads
    policy_flops = 2 * S * D * 6 + 2 * D * 6
    value_sv_flops = 2 * S * D * 29
    num_scorebeliefs = config["num_scorebeliefs"]
    scorebelief_len = (S + EXTRA_SCORE_DISTR_RADIUS) * 2
    score_mix_out = scorebelief_len * num_scorebeliefs + num_scorebeliefs
    score_flops = 2 * D * score_mix_out

    total = trunk_flops + conv_flops + global_flops + policy_flops + value_sv_flops + score_flops
    return total


def get_gpu_peak_tflops(device):
    """Return BF16 peak TFLOPS for MFU calculation."""
    if device.type != "cuda":
        return 0.0

    name = torch.cuda.get_device_name(device).lower()
    known_gpus = {
        "4090": 165.2,
        "4080 super": 97.5,
        "4080": 97.5,
        "4070 ti super": 93.2,
        "4070 ti": 40.1,
        "4070": 29.1,
        "3090 ti": 40.0,
        "3090": 35.6,
        "3080 ti": 34.1,
        "3080": 29.8,
        "a100 sxm": 312.0,
        "a100 pcie": 312.0,
        "a100": 312.0,
        "a6000": 38.7,
        "a10": 31.2,
        "h100 sxm": 989.5,
        "h100 pcie": 756.0,
        "h100": 756.0,
        "h200": 989.5,
        "l40s": 91.6,
        "l40": 90.5,
        "l4": 30.3,
    }
    for key, tflops in known_gpus.items():
        if key in name:
            return tflops

    props = torch.cuda.get_device_properties(device)
    clock_ghz = props.clock_rate / 1e6
    estimated = props.multi_processor_count * 128 * 2 * clock_ghz / 1e3
    logging.warning(
        f"Unknown GPU '{torch.cuda.get_device_name(device)}', "
        f"rough BF16 estimate: {estimated:.1f} TFLOPS (MFU may be inaccurate)"
    )
    return estimated
