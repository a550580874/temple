#!/usr/bin/env python3
import argparse
import copy
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import torch


OPTIM_KEYS = ("param", "exp_avg", "exp_avg_sq")
PARAM_SUFFIXES = (
    "layer_norm_weight",
    "layer_norm_bias",
    "weight",
    "bias",
)


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="How many target distrib_optim.pt files to rewrite in parallel.",
    )
    parser.add_argument(
        "--path-contains",
        action="append",
        default=[],
        help=(
            "Only process distrib_optim.pt paths whose relative path contains this string. "
            "Can be provided multiple times, e.g. --path-contains mp_rank_00_001."
        ),
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


def _module_prefix(param_name):
    for suffix in PARAM_SUFFIXES:
        token = f".{suffix}"
        if param_name.endswith(token):
            return param_name[: -len(token)]
    raise ValueError(f"Unsupported parameter name for module rewrite: {param_name}")


def _discover_optimizer_files(root_dir):
    paths = []
    for current_root, _dirs, files in os.walk(root_dir):
        if "distrib_optim.pt" in files:
            paths.append(os.path.join(current_root, "distrib_optim.pt"))
    return sorted(paths)


def _filter_optimizer_paths(paths, root_dir, filters):
    if not filters:
        return paths
    selected = []
    for path in paths:
        rel_path = os.path.relpath(path, root_dir)
        if any(token in rel_path for token in filters):
            selected.append(path)
    return selected


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


def _sort_entries_for_layout(entries):
    return sorted(
        entries,
        key=lambda entry: (
            entry["rank"],
            entry["gbuf_idx"],
            entry["bucket_idx"],
            entry["gbuf_world"]["start"],
            entry["gbuf_world"]["end"],
            entry["param_name"],
        ),
    )


def _build_module_rewrite_plan(changed_pairs, old_param_map, new_param_map):
    changed_modules = {_module_prefix(param_name) for param_name, _, _ in changed_pairs}
    module_plans = []
    for module_prefix in sorted(changed_modules):
        old_entries = _sort_entries_for_layout(
            [entry for name, entry in old_param_map.items() if _module_prefix(name) == module_prefix]
        )
        new_entries = _sort_entries_for_layout(
            [entry for name, entry in new_param_map.items() if _module_prefix(name) == module_prefix]
        )
        old_total = sum(entry["gbuf_world"]["size"] for entry in old_entries)
        new_total = sum(entry["gbuf_world"]["size"] for entry in new_entries)
        if old_total != new_total:
            raise ValueError(
                f"Module {module_prefix} changed total size from {old_total} to {new_total}, "
                "which this converter does not support."
            )
        module_plans.append(
            {
                "module_prefix": module_prefix,
                "old_entries": old_entries,
                "new_entries": new_entries,
                "total_size": old_total,
            }
        )
    return module_plans


def _rewrite_target_rank(
    target_rank,
    target_path,
    rank_to_modules,
    input_optim_paths,
):
    target_state = copy.deepcopy(torch.load(target_path, map_location="cpu"))
    source_state_cache = {}

    for module_plan in rank_to_modules[target_rank]:
        for tensor_key in OPTIM_KEYS:
            source_chunks = []
            for old_entry in module_plan["old_entries"]:
                source_rank = old_entry["rank"]
                source_path = input_optim_paths[source_rank]
                if source_rank not in source_state_cache:
                    source_state_cache[source_rank] = torch.load(source_path, map_location="cpu")
                source_state = source_state_cache[source_rank]
                source_tensor = _get_world_tensor(source_state, old_entry, tensor_key)
                start = old_entry["gbuf_world"]["start"]
                end = old_entry["gbuf_world"]["end"]
                source_chunks.append(source_tensor[start:end].clone())

            module_blob = torch.cat(source_chunks, dim=0)
            cursor = 0

            for new_entry in module_plan["new_entries"]:
                target_start = new_entry["gbuf_world"]["start"]
                target_end = new_entry["gbuf_world"]["end"]
                chunk_size = target_end - target_start
                if new_entry["rank"] == target_rank:
                    target_tensor = _get_world_tensor(target_state, new_entry, tensor_key)
                    target_tensor[target_start:target_end].copy_(
                        module_blob[cursor : cursor + chunk_size]
                    )
                cursor += chunk_size

            if cursor != module_blob.numel():
                raise ValueError(
                    f"Module {module_plan['module_prefix']} key={tensor_key} "
                    f"consumed {cursor} elements but source blob has {module_blob.numel()}."
                )

    torch.save(target_state, target_path)
    return target_path


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

    module_plans = _build_module_rewrite_plan(changed_pairs, old_param_map, new_param_map)

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
    summary["changed_module_count"] = len(module_plans)
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

    selected_output_paths = _filter_optimizer_paths(output_optim_paths, args.output_root, args.path_contains)
    if args.path_contains and not selected_output_paths:
        raise ValueError(
            f"No distrib_optim.pt matched --path-contains filters: {args.path_contains}"
        )

    selected_target_ranks = {
        rank for rank, path in enumerate(output_optim_paths) if path in selected_output_paths
    }
    rank_to_modules = defaultdict(list)
    for module_plan in module_plans:
        for rank in {entry["rank"] for entry in module_plan["new_entries"]}:
            if rank in selected_target_ranks:
                rank_to_modules[rank].append(module_plan)

    if not rank_to_modules:
        print("No target ranks matched the selected filters.")
        return

    workers = max(1, args.workers)
    tasks = [
        (target_rank, output_optim_paths[target_rank], rank_to_modules, input_optim_paths)
        for target_rank in sorted(rank_to_modules)
    ]

    if workers == 1:
        for task in tasks:
            target_path = _rewrite_target_rank(*task)
            print(f"[write] rewrote {target_path}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_rewrite_target_rank, *task) for task in tasks]
            for future in futures:
                target_path = future.result()
                print(f"[write] rewrote {target_path}")


if __name__ == "__main__":
    main()
