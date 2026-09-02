#!/usr/bin/env python3
"""Generate a standalone DeepEP replay script from DEEPEP_OP_CAPTURE output.

The generated replay uses the captured routing tensors and per-rank input
shapes. It intentionally replays each dispatch followed by its matching
combine; it is a kernel/integration reproducer, not a model correctness test.
"""

import argparse
import json
from pathlib import Path


def read_records(root: Path):
    records = []
    for path in sorted(root.glob("rank_*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("phase") == "before":
                records.append(item)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("replay_deepep.py"))
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--num-max-dispatch-tokens-per-rank", type=int, default=256)
    args = parser.parse_args()

    records = read_records(args.capture_dir)
    if not records:
        raise SystemExit(f"no capture records found in {args.capture_dir}")

    # Keep only the latest record for overwritten ring-buffer slots.
    latest = {}
    for item in records:
        key = (item["rank"], item.get("slot", item["call_id"]))
        if item["call_id"] >= latest.get(key, {}).get("call_id", -1):
            latest[key] = item
    by_rank = {}
    for item in latest.values():
        by_rank.setdefault(item["rank"], []).append(item)
    for items in by_rank.values():
        items.sort(key=lambda x: x["call_id"])

    # Embed only metadata; tensor files remain alongside the generated script.
    payload = []
    for rank in range(args.num_processes):
        rank_items = []
        for item in by_rank.get(rank, []):
            if item["op"] not in ("normal_dispatch", "low_latency_dispatch"):
                continue
            item = dict(item)
            item["tensor_files"] = {
                k: str((args.capture_dir / v).resolve())
                for k, v in item.get("tensor_files", {}).items()
            }
            rank_items.append(item)
        payload.append(rank_items)

    generated = f'''#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import deep_ep

NUM_PROCESSES = {args.num_processes}
NUM_EXPERTS = {args.num_experts}
NUM_MAX = {args.num_max_dispatch_tokens_per_rank}
CAPTURE = {repr(payload)}

def init_rank(local_rank):
    addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = os.getenv("MASTER_PORT", "8361")
    node_rank = int(os.getenv("RANK", "0"))
    nodes = int(os.getenv("WORLD_SIZE", "1"))
    rank = node_rank * NUM_PROCESSES + local_rank
    world = nodes * NUM_PROCESSES
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method=f"tcp://{{addr}}:{{port}}", rank=rank, world_size=world)
    return dist.new_group(list(range(world))), rank

def run(local_rank):
    group, rank = init_rank(local_rank)
    # AUTO uses a low-latency-capable buffer, matching SGLang's mixed setup.
    buffer = deep_ep.Buffer(group, int(2e9), 0, low_latency_mode=True, num_qps_per_rank=NUM_EXPERTS // NUM_PROCESSES)
    for item in CAPTURE[rank]:
        op = item["op"]
        x_shape = tuple(item["x_shape"])
        topk = torch.load(item["tensor_files"]["topk_idx"], map_location="cpu").to("npu")
        weights_path = item["tensor_files"].get("topk_weights")
        weights = torch.load(weights_path, map_location="cpu").to("npu") if weights_path else None
        x = torch.randn(x_shape, dtype=torch.bfloat16, device="npu")
        if op == "normal_dispatch":
            layout = buffer.get_dispatch_layout(topk, NUM_EXPERTS)
            recv_x, _, _, _, handle, _ = buffer.dispatch(
                x=x, num_tokens_per_rank=layout[0], num_tokens_per_rdma_rank=layout[1],
                num_tokens_per_expert=layout[2], is_token_in_rank=layout[3], topk_idx=topk,
                topk_weights=weights, async_finish=item["async_finish"],
                allocate_on_comm_stream=item["allocate_on_comm_stream"])
            combine_x = recv_x[0] if isinstance(recv_x, tuple) else recv_x
            recv_weights = handle[-1]
            buffer.combine(x=combine_x, handle=handle, topk_weights=recv_weights,
                           async_finish=item["async_finish"], allocate_on_comm_stream=item["allocate_on_comm_stream"])
        else:
            num_max = item.get("num_max_dispatch_tokens_per_rank", NUM_MAX)
            recv_x, _, handle, _, _ = buffer.low_latency_dispatch(
                x, topk, num_max, NUM_EXPERTS,
                use_fp8=item.get("use_fp8", False),
                use_ue8m0=item.get("use_ue8m0", False),
                use_mxfp4=item.get("use_mxfp4", False),
                async_finish=item["async_finish"], return_recv_hook=item["return_recv_hook"])
            combine_x = recv_x[0] if isinstance(recv_x, tuple) else recv_x
            if weights is None:
                weights = torch.ones_like(topk, dtype=torch.float32, device="npu")
            buffer.low_latency_combine(combine_x, topk, weights, handle,
                                       async_finish=item["async_finish"], return_recv_hook=item["return_recv_hook"])
    torch.npu.synchronize()
    dist.barrier(group=group)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(run, nprocs=NUM_PROCESSES)
'''
    args.output.write_text(generated)
    print(f"wrote {args.output} with {len(records)} captured records")


if __name__ == "__main__":
    main()
