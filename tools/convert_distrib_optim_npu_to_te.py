#!/usr/bin/env python3
import argparse
import copy
import json
import os
import shutil
from collections import defaultdict

import torch


OPTIM_KEYS = ("param", "exp_avg", "exp_avg_sq")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert legacy distributed optimizer checkpoints from MindSpeed's old "
            "TELayerNormColumnParallelLinear ordering to TE/Megatron ordering."
        )
    )
    parser.add_argument("--input-root", required=True, help="Input checkpoint root directory.")
    parser.add_argument("--output-root", required=True, help="Output checkpoint root directory.")
    parser.add_argument("--old-dump-dir", required=True, help="Full old bucket-map dump directory.")
    parser.add_argument("--new-dump-dir", required=True, help="Full new bucket-map dump directory.")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Remove output-root first if it already exists.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to mirror non-optimizer files into output-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate and print the rewrite plan without writing files.",
    )
    return parser.parse_args()


def _load_dump_dir(dump_dir):
    rank_to_entries = {}
    param_to_entry = {}

    for filename in sorted(os.listdir(dump_dir)):
        if not filename.startswith("optimizer_bucket_map_rank") or not filename.endswith(".json"):
            continue
        path = os.path.join(dump_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rank = payload["rank"]
        entries = []
        for optimizer in payload.get("optimizers", []):
            for entry in optimizer.get("entries", []):
                full_entry = {
                    "rank": rank,
                    **entry,
                }
                entries.append(full_entry)
                param_to_entry[entry["param_name"]] = full_entry
        rank_to_entries[rank] = entries

    return rank_to_entries, param_to_entry


def _entry_signature(entry):
    return (
        entry["rank"],
        entry["gbuf_idx"],
        entry["bucket_idx"],
        entry["dtype"],
        entry["gbuf_world"]["start"],
        entry["gbuf_world"]["end"],
    )


def _discover_optimizer_files(root_dir):
    paths = []
    for current_root, _dirs, files in os.walk(root_dir):
        if "distrib_optim.pt" in files:
            paths.append(os.path.join(current_root, "distrib_optim.pt"))
    return sorted(paths)


def _hardlink_or_copy(src, dst, mode):
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _mirror_tree(input_root, output_root, mode):
    for current_root, dirs, files in os.walk(input_root):
        rel_root = os.path.relpath(current_root, input_root)
        target_root = output_root if rel_root == "." else os.path.join(output_root, rel_root)
        os.makedirs(target_root, exist_ok=True)
        for directory in dirs:
            os.makedirs(os.path.join(target_root, directory), exist_ok=True)
        for filename in files:
            src = os.path.join(current_root, filename)
            dst = os.path.join(target_root, filename)
            _hardlink_or_copy(src, dst, mode)


def _get_dtype_key(dtype_map, dtype_string):
    for key in dtype_map.keys():
        if str(key) == dtype_string:
            return key
    raise KeyError(f"Unable to find dtype key {dtype_string!r} in {list(dtype_map.keys())}")


def _get_world_tensor(state_dict, entry, tensor_key):
    gbuf_state = state_dict[entry["gbuf_idx"]]
    dtype_key = _get_dtype_key(gbuf_state, entry["dtype"])
    return gbuf_state[dtype_key][tensor_key]


def _validate_entry_pair(param_name, old_entry, new_entry):
    old_size = old_entry["gbuf_world"]["size"]
    new_size = new_entry["gbuf_world"]["size"]
    if old_size != new_size:
        raise ValueError(
            f"Parameter {param_name} changed size from {old_size} to {new_size}, "
            "which this converter does not support."
        )
    for field in ("optimizer_index", "gbuf_idx", "bucket_idx", "dtype"):
        if field == "optimizer_index":
            # The checkpoint file contains one distrib_optim.pt per rank/shard; optimizer_index
            # is metadata only and does not affect the tensor layout in distrib_optim.pt.
            continue
        if old_entry[field] != new_entry[field]:
            raise ValueError(
                f"Parameter {param_name} changed {field} from {old_entry[field]!r} "
                f"to {new_entry[field]!r}; unsupported without a format-specific migration."
            )


def _summarize_plan(changed_pairs):
    per_target_rank = defaultdict(int)
    moved_across_rank = 0
    for _name, old_entry, new_entry in changed_pairs:
        per_target_rank[new_entry["rank"]] += 1
        if old_entry["rank"] != new_entry["rank"]:
            moved_across_rank += 1
    return {
        "changed_param_count": len(changed_pairs),
        "cross_rank_moves": moved_across_rank,
        "target_ranks_touched": len(per_target_rank),
    }


def main():
    args = parse_args()

    if os.path.abspath(args.input_root) == os.path.abspath(args.output_root):
        raise ValueError("--input-root and --output-root must be different.")

    old_rank_entries, old_param_map = _load_dump_dir(args.old_dump_dir)
    new_rank_entries, new_param_map = _load_dump_dir(args.new_dump_dir)

    old_names = set(old_param_map)
    new_names = set(new_param_map)
    if old_names != new_names:
        only_old = sorted(old_names - new_names)[:10]
        only_new = sorted(new_names - old_names)[:10]
        raise ValueError(
            "Old/new dump parameter sets differ. "
            f"only_old(sample)={only_old}, only_new(sample)={only_new}"
        )

    changed_pairs = []
    for param_name in sorted(old_names):
        old_entry = old_param_map[param_name]
        new_entry = new_param_map[param_name]
        if _entry_signature(old_entry) != _entry_signature(new_entry):
            _validate_entry_pair(param_name, old_entry, new_entry)
            changed_pairs.append((param_name, old_entry, new_entry))

    if not changed_pairs:
        print("No distributed optimizer slice changes detected between old/new dumps.")
        return

    input_optim_paths = _discover_optimizer_files(args.input_root)
    if not input_optim_paths:
        raise FileNotFoundError(f"No distrib_optim.pt files found under {args.input_root}")

    rank_count = max(max(old_rank_entries), max(new_rank_entries)) + 1
    if len(input_optim_paths) != rank_count:
        raise ValueError(
            f"Found {len(input_optim_paths)} distrib_optim.pt files but dump ranks span 0..{rank_count - 1}. "
            "This script currently assumes one distrib_optim.pt per dump rank."
        )

    summary = _summarize_plan(changed_pairs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for param_name, old_entry, new_entry in changed_pairs:
        if old_entry["rank"] != new_entry["rank"]:
            print(
                f"[move] {param_name}: rank {old_entry['rank']} -> {new_entry['rank']}, "
                f"{old_entry['gbuf_world']} -> {new_entry['gbuf_world']}"
            )

    if args.dry_run:
        return

    if os.path.exists(args.output_root):
        if not args.overwrite_output:
            raise FileExistsError(
                f"Output root already exists: {args.output_root}. "
                "Use --overwrite-output to replace it."
            )
        shutil.rmtree(args.output_root)

    _mirror_tree(args.input_root, args.output_root, args.copy_mode)

    output_optim_paths = _discover_optimizer_files(args.output_root)
    if len(output_optim_paths) != len(input_optim_paths):
        raise ValueError("Output checkpoint tree does not contain the expected distrib_optim.pt files.")

    source_state_cache = {}
    target_ops = defaultdict(list)
    for param_name, old_entry, new_entry in changed_pairs:
        target_ops[new_entry["rank"]].append((param_name, old_entry, new_entry))

    for target_rank, operations in sorted(target_ops.items()):
        target_path = output_optim_paths[target_rank]
        target_state = torch.load(target_path, map_location="cpu")
        target_state = copy.deepcopy(target_state)

        for param_name, old_entry, new_entry in operations:
            source_rank = old_entry["rank"]
            source_path = input_optim_paths[source_rank]
            if source_rank not in source_state_cache:
                source_state_cache[source_rank] = torch.load(source_path, map_location="cpu")
            source_state = source_state_cache[source_rank]

            for tensor_key in OPTIM_KEYS:
                source_tensor = _get_world_tensor(source_state, old_entry, tensor_key)
                target_tensor = _get_world_tensor(target_state, new_entry, tensor_key)

                source_start = old_entry["gbuf_world"]["start"]
                source_end = old_entry["gbuf_world"]["end"]
                target_start = new_entry["gbuf_world"]["start"]
                target_end = new_entry["gbuf_world"]["end"]

                source_view = source_tensor[source_start:source_end]
                target_view = target_tensor[target_start:target_end]
                if source_view.numel() != target_view.numel():
                    raise ValueError(
                        f"Slice size mismatch for {param_name} key={tensor_key}: "
                        f"{source_view.numel()} vs {target_view.numel()}"
                    )
                target_view.copy_(source_view)

        torch.save(target_state, target_path)
        print(f"[write] rewrote {target_path}")


if __name__ == "__main__":
    main()
