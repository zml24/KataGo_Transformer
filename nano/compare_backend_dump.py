#!/usr/bin/env python3
"""Compare KataGo backend Transformer dump against torch reference."""

import argparse
import json
import math

import torch
import torch.nn.functional as F

from export_cuda import POLICY_CHANNELS, _make_model
from model import apply_rotary_emb


def _torch_inputs_from_sample(sample, pos_len, num_spatial_features):
    spatial = torch.tensor(sample["spatialNHWC"], dtype=torch.float32)
    spatial = spatial.view(pos_len, pos_len, num_spatial_features)
    global_input = torch.tensor(sample["global"], dtype=torch.float32).unsqueeze(0)
    return spatial, global_input


def _apply_input_symmetry(spatial_nhwc, symmetry):
    if symmetry & 0x1:
        spatial_nhwc = torch.flip(spatial_nhwc, dims=[0])
    if symmetry & 0x2:
        spatial_nhwc = torch.flip(spatial_nhwc, dims=[1])
    if symmetry & 0x4:
        spatial_nhwc = torch.transpose(spatial_nhwc, 0, 1)
    return spatial_nhwc


def _invert_spatial_outputs(flat_values, pos_len, channels, symmetry):
    tensor = torch.tensor(flat_values, dtype=torch.float32).view(pos_len, pos_len, channels)
    if symmetry & 0x4:
        tensor = torch.transpose(tensor, 0, 1)
    if symmetry & 0x2:
        tensor = torch.flip(tensor, dims=[1])
    if symmetry & 0x1:
        tensor = torch.flip(tensor, dims=[0])
    return tensor.contiguous().view(-1, channels)


def _flatten_spatial_nchw(tensor):
    return tensor.permute(1, 2, 0).contiguous().view(-1, tensor.shape[0]).flatten().tolist()


def _extract_export_subset(outputs, pos_len, policy_optimism, symmetry):
    out_policy, out_value, out_misc, out_moremisc, out_ownership, *_rest = outputs
    seq_len = pos_len * pos_len

    policy = out_policy[0, POLICY_CHANNELS, :seq_len].transpose(0, 1).contiguous()
    policy_pass = out_policy[0, POLICY_CHANNELS, seq_len].contiguous()
    score_value = torch.cat([out_misc[0, 0:4], out_moremisc[0, 0:2]], dim=0).contiguous()
    ownership = out_ownership[0, 0].flatten().contiguous()

    policy_probs = policy[:, 0] + (policy[:, 1] - policy[:, 0]) * policy_optimism
    pass_prob = policy_pass[0] + (policy_pass[1] - policy_pass[0]) * policy_optimism
    policy_probs = _invert_spatial_outputs(policy_probs.tolist(), pos_len, 1, symmetry).view(-1)
    ownership_nn = _invert_spatial_outputs(ownership.tolist(), pos_len, 1, symmetry).view(-1)

    return {
        "raw": {
            "policy": policy.flatten().tolist(),
            "policyPass": policy_pass.tolist(),
            "value": out_value[0].contiguous().tolist(),
            "scoreValue": score_value.tolist(),
            "ownership": ownership.tolist(),
        },
        "nnOutput": {
            "policyProbs": torch.cat([policy_probs, pass_prob.view(1)], dim=0).tolist(),
            "whiteWinProb": float(out_value[0, 0]),
            "whiteLossProb": float(out_value[0, 1]),
            "whiteNoResultProb": float(out_value[0, 2]),
            "whiteScoreMean": float(score_value[0]),
            "whiteScoreMeanSq": float(score_value[1]),
            "whiteLead": float(score_value[2]),
            "varTimeLeft": float(score_value[3]),
            "shorttermWinlossError": float(score_value[4]),
            "shorttermScoreError": float(score_value[5]),
            "whiteOwnerMap": ownership_nn.tolist(),
        },
    }


def _extract_full_raw(outputs, pos_len):
    out_policy, out_value, out_misc, out_moremisc, out_ownership, out_scoring, out_futurepos, out_seki, out_scorebelief = outputs
    seq_len = pos_len * pos_len
    policy = out_policy[0, :, :seq_len].transpose(0, 1).contiguous()
    policy_pass = out_policy[0, :, seq_len].contiguous()
    return {
        "policy": policy.flatten().tolist(),
        "policyPass": policy_pass.tolist(),
        "value": out_value[0].contiguous().tolist(),
        "misc": out_misc[0].contiguous().tolist(),
        "moreMisc": out_moremisc[0].contiguous().tolist(),
        "ownership": _flatten_spatial_nchw(out_ownership[0]),
        "scoring": _flatten_spatial_nchw(out_scoring[0]),
        "futurePos": _flatten_spatial_nchw(out_futurepos[0]),
        "seki": _flatten_spatial_nchw(out_seki[0]),
        "scoreBelief": out_scorebelief[0].contiguous().tolist(),
    }


def _extract_postprocess(outputs, post_outputs):
    (
        _policy_logits,
        _value_logits,
        td_value_logits,
        pred_td_score,
        _ownership_pretanh,
        _pred_scoring,
        _futurepos_pretanh,
        _seki_logits,
        pred_scoremean,
        pred_scorestdev,
        pred_lead,
        pred_variance_time,
        pred_shortterm_value_error,
        pred_shortterm_score_error,
        _scorebelief_logits,
    ) = post_outputs
    return {
        "tdValueLogits": td_value_logits[0].contiguous().flatten().tolist(),
        "predTdScore": pred_td_score[0].contiguous().tolist(),
        "predScoreMean": float(pred_scoremean[0]),
        "predScoreStdev": float(pred_scorestdev[0]),
        "predLead": float(pred_lead[0]),
        "predVarianceTime": float(pred_variance_time[0]),
        "predShorttermValueError": float(pred_shortterm_value_error[0]),
        "predShorttermScoreError": float(pred_shortterm_score_error[0]),
    }


def _run_block_explicit(block, x, rope_cos=None, rope_sin=None):
    batch_size, seq_len, hidden = x.shape
    x_normed = block.norm1(x)
    q = block.q_proj(x_normed).view(batch_size, seq_len, block.num_heads, block.head_dim)
    k = block.k_proj(x_normed).view(batch_size, seq_len, block.num_heads, block.head_dim)
    v = block.v_proj(x_normed).view(batch_size, seq_len, block.num_heads, block.head_dim)

    if rope_cos is not None:
        q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(block.head_dim))
    probs = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(probs, v)
    attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden)
    x = x + block.out_proj(attn_out)

    x_normed = block.norm2(x)
    x = x + block.ffn_w2(F.silu(block.ffn_w1(x_normed)) * block.ffn_wgate(x_normed))
    return x


def _run_model_outputs(model, spatial, global_input, attention_mode):
    if attention_mode == "default":
        outputs = model(spatial, global_input)
        post_outputs = model.postprocess(outputs)
        return outputs, post_outputs
    if attention_mode != "explicit":
        raise ValueError(f"unknown attention mode: {attention_mode}")

    x = model._forward_stem_impl(spatial, global_input)
    for block in model.blocks:
        x = _run_block_explicit(block, x, model.rope_cos, model.rope_sin)
    x = model.norm_final(x)
    outputs = (
        model.policy_head(x).float(),
        *[tensor.float() for tensor in model.value_head(x, global_input[:, -1:])],
    )
    post_outputs = model.postprocess(outputs)
    return outputs, post_outputs


def _softplus(x):
    x = float(x)
    if x > 20.0:
        return x
    return math.log1p(math.exp(x))


def _backend_postprocess_from_raw_full(raw_full, params):
    misc = raw_full["misc"]
    moremisc = raw_full["moreMisc"]
    return {
        "tdValueLogits": list(misc[4:7]) + list(misc[7:10]) + list(moremisc[2:5]),
        "predTdScore": [float(x) * float(params["tdScoreMultiplier"]) for x in moremisc[5:8]],
        "predScoreMean": float(misc[0]) * float(params["scoreMeanMultiplier"]),
        "predScoreStdev": _softplus(misc[1]) * float(params["scoreStdevMultiplier"]),
        "predLead": float(misc[2]) * float(params["leadMultiplier"]),
        "predVarianceTime": _softplus(misc[3]) * float(params["varianceTimeMultiplier"]),
        "predShorttermValueError": (_softplus(0.5 * float(moremisc[0])) ** 2) * float(params["shorttermValueErrorMultiplier"]),
        "predShorttermScoreError": (_softplus(0.5 * float(moremisc[1])) ** 2) * float(params["shorttermScoreErrorMultiplier"]),
    }


def _max_abs_err(lhs, rhs):
    if len(lhs) != len(rhs):
        raise ValueError(f"length mismatch: {len(lhs)} vs {len(rhs)}")
    if not lhs:
        return 0.0
    return max(abs(float(a) - float(b)) for a, b in zip(lhs, rhs))


def _compare_sample(sample_name, backend_sample, torch_sample):
    report = {
        "name": sample_name,
        "raw": {
            "policy": _max_abs_err(backend_sample["raw"]["policy"], torch_sample["raw"]["policy"]),
            "policyPass": _max_abs_err(backend_sample["raw"]["policyPass"], torch_sample["raw"]["policyPass"]),
            "value": _max_abs_err(backend_sample["raw"]["value"], torch_sample["raw"]["value"]),
            "scoreValue": _max_abs_err(backend_sample["raw"]["scoreValue"], torch_sample["raw"]["scoreValue"]),
            "ownership": _max_abs_err(backend_sample["raw"]["ownership"], torch_sample["raw"]["ownership"]),
        },
        "nnOutput": {
            "policyProbs": _max_abs_err(backend_sample["nnOutput"]["policyProbs"], torch_sample["nnOutput"]["policyProbs"]),
            "whiteOwnerMap": _max_abs_err(backend_sample["nnOutput"]["whiteOwnerMap"], torch_sample["nnOutput"]["whiteOwnerMap"]),
            "whiteWinProb": abs(float(backend_sample["nnOutput"]["whiteWinProb"]) - float(torch_sample["nnOutput"]["whiteWinProb"])),
            "whiteLossProb": abs(float(backend_sample["nnOutput"]["whiteLossProb"]) - float(torch_sample["nnOutput"]["whiteLossProb"])),
            "whiteNoResultProb": abs(float(backend_sample["nnOutput"]["whiteNoResultProb"]) - float(torch_sample["nnOutput"]["whiteNoResultProb"])),
            "whiteScoreMean": abs(float(backend_sample["nnOutput"]["whiteScoreMean"]) - float(torch_sample["nnOutput"]["whiteScoreMean"])),
            "whiteScoreMeanSq": abs(float(backend_sample["nnOutput"]["whiteScoreMeanSq"]) - float(torch_sample["nnOutput"]["whiteScoreMeanSq"])),
            "whiteLead": abs(float(backend_sample["nnOutput"]["whiteLead"]) - float(torch_sample["nnOutput"]["whiteLead"])),
            "varTimeLeft": abs(float(backend_sample["nnOutput"]["varTimeLeft"]) - float(torch_sample["nnOutput"]["varTimeLeft"])),
            "shorttermWinlossError": abs(float(backend_sample["nnOutput"]["shorttermWinlossError"]) - float(torch_sample["nnOutput"]["shorttermWinlossError"])),
            "shorttermScoreError": abs(float(backend_sample["nnOutput"]["shorttermScoreError"]) - float(torch_sample["nnOutput"]["shorttermScoreError"])),
        },
    }
    if "rawFull" in backend_sample and "rawFull" in torch_sample:
        report["rawFull"] = {
            "policy": _max_abs_err(backend_sample["rawFull"]["policy"], torch_sample["rawFull"]["policy"]),
            "policyPass": _max_abs_err(backend_sample["rawFull"]["policyPass"], torch_sample["rawFull"]["policyPass"]),
            "value": _max_abs_err(backend_sample["rawFull"]["value"], torch_sample["rawFull"]["value"]),
            "misc": _max_abs_err(backend_sample["rawFull"]["misc"], torch_sample["rawFull"]["misc"]),
            "moreMisc": _max_abs_err(backend_sample["rawFull"]["moreMisc"], torch_sample["rawFull"]["moreMisc"]),
            "ownership": _max_abs_err(backend_sample["rawFull"]["ownership"], torch_sample["rawFull"]["ownership"]),
            "scoring": _max_abs_err(backend_sample["rawFull"]["scoring"], torch_sample["rawFull"]["scoring"]),
            "futurePos": _max_abs_err(backend_sample["rawFull"]["futurePos"], torch_sample["rawFull"]["futurePos"]),
            "seki": _max_abs_err(backend_sample["rawFull"]["seki"], torch_sample["rawFull"]["seki"]),
            "scoreBelief": _max_abs_err(backend_sample["rawFull"]["scoreBelief"], torch_sample["rawFull"]["scoreBelief"]),
        }
    if "postprocess" in backend_sample and "postprocess" in torch_sample:
        report["postprocess"] = {
            "tdValueLogits": _max_abs_err(backend_sample["postprocess"]["tdValueLogits"], torch_sample["postprocess"]["tdValueLogits"]),
            "predTdScore": _max_abs_err(backend_sample["postprocess"]["predTdScore"], torch_sample["postprocess"]["predTdScore"]),
            "predScoreMean": abs(float(backend_sample["postprocess"]["predScoreMean"]) - float(torch_sample["postprocess"]["predScoreMean"])),
            "predScoreStdev": abs(float(backend_sample["postprocess"]["predScoreStdev"]) - float(torch_sample["postprocess"]["predScoreStdev"])),
            "predLead": abs(float(backend_sample["postprocess"]["predLead"]) - float(torch_sample["postprocess"]["predLead"])),
            "predVarianceTime": abs(float(backend_sample["postprocess"]["predVarianceTime"]) - float(torch_sample["postprocess"]["predVarianceTime"])),
            "predShorttermValueError": abs(float(backend_sample["postprocess"]["predShorttermValueError"]) - float(torch_sample["postprocess"]["predShorttermValueError"])),
            "predShorttermScoreError": abs(float(backend_sample["postprocess"]["predShorttermScoreError"]) - float(torch_sample["postprocess"]["predShorttermScoreError"])),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="比较 KataGo backend dump 和 torch Transformer 参考结果")
    parser.add_argument("--checkpoint", required=True, help="checkpoint.ckpt 路径")
    parser.add_argument("--dump-json", required=True, help="katago runtransformerdump 生成的 JSON")
    parser.add_argument("--output-json", help="可选，写出对照报告 JSON")
    parser.add_argument("--score-mode", default="mixop", choices=["simple", "mix", "mixop"])
    parser.add_argument("--use-ema", action="store_true", help="优先使用 EMA 权重")
    parser.add_argument("--attention-mode", default="default", choices=["default", "explicit"])
    args = parser.parse_args()

    with open(args.dump_json, "r", encoding="utf-8") as f:
      backend_dump = json.load(f)

    pos_len = int(backend_dump["nnXLen"])
    model, config = _make_model(args.checkpoint, pos_len, args.score_mode, args.use_ema)
    model.eval()
    num_spatial_features = len(backend_dump["samples"][0]["spatialNHWC"]) // (pos_len * pos_len)
    policy_optimism = float(backend_dump.get("policyOptimism", 0.0))

    postprocess_params = backend_dump.get("postProcessParams", {
        "tdScoreMultiplier": 20.0,
        "scoreMeanMultiplier": 20.0,
        "scoreStdevMultiplier": 20.0,
        "leadMultiplier": 20.0,
        "varianceTimeMultiplier": 40.0,
        "shorttermValueErrorMultiplier": 0.25,
        "shorttermScoreErrorMultiplier": 150.0,
    })

    torch_samples = []
    with torch.no_grad():
        for sample in backend_dump["samples"]:
            spatial, global_input = _torch_inputs_from_sample(sample, pos_len, num_spatial_features)
            spatial = _apply_input_symmetry(spatial, int(backend_dump.get("symmetry", 0)))
            spatial = spatial.permute(2, 0, 1).unsqueeze(0).contiguous()
            outputs, post_outputs = _run_model_outputs(model, spatial, global_input, args.attention_mode)
            torch_sample = _extract_export_subset(outputs, pos_len, policy_optimism, int(backend_dump.get("symmetry", 0)))
            if int(backend_dump.get("numFullPolicyChannels", 0)) > 0:
                torch_sample["rawFull"] = _extract_full_raw(outputs, pos_len)
                torch_sample["postprocess"] = _extract_postprocess(outputs, post_outputs)
            torch_samples.append(torch_sample)

    sample_reports = []
    global_max = 0.0
    for backend_sample, torch_sample in zip(backend_dump["samples"], torch_samples):
        if "rawFull" in backend_sample:
            backend_sample = dict(backend_sample)
            backend_sample["postprocess"] = _backend_postprocess_from_raw_full(backend_sample["rawFull"], postprocess_params)
        report = _compare_sample(backend_sample["name"], backend_sample, torch_sample)
        sample_reports.append(report)
        for group in report.values():
            if isinstance(group, dict):
                for value in group.values():
                    if isinstance(value, dict):
                        for sub_value in value.values():
                            global_max = max(global_max, float(sub_value))
                    else:
                        global_max = max(global_max, float(value))

    result = {
        "checkpoint": args.checkpoint,
        "dumpJson": args.dump_json,
        "modelVersion": config["version"],
        "posLen": pos_len,
        "attentionMode": args.attention_mode,
        "maxAbsErr": global_max,
        "samples": sample_reports,
    }

    text = json.dumps(result, indent=2)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
