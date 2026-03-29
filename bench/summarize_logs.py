# ## What gets logged per run
# Performance:    wall_time, total_tokens, agg_throughput, TTFT avg/p50/p99, TPOT avg
# vLLM metrics:   kv_cache_usage_perc  (min/max/avg sampled every 0.5s during run)
#                 num_requests_running (min/max/avg)
#                 num_requests_waiting (min/max/avg)  ← tells you when queue built up
#                 num_requests_swapped (min/max/avg)  ← requests evicted from KV cache
# Per-request:    full TTFT/TPOT/tokens for every individual request in the run



import json
import sys
from collections import defaultdict

LOG_FILE = "logs/bench_results.jsonl"

def load_logs(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def summarize(records):

    # Only summarize averaged records, not individual trials
    records = [r for r in records if r.get("record_type") == "averaged_summary"]

    if not records:
        print("No averaged_summary records found. Have you run run_experiment() yet?")
        return
    
    by_tag = defaultdict(list)
    for r in records:
        by_tag[r["tag"]].append(r)

    for tag, runs in by_tag.items():
        print(f"\n{'='*80}")
        print(f"Tag: {tag}  ({len(runs)} runs recorded)")
        print(f"{'='*80}")

        # Performance table
        print(f"\n{'n':>5}  {'tok/s':>7}  {'TTFT_avg':>9}  {'TTFT_p99':>9}  {'TPOT_avg':>9}  {'wall_s':>6}")
        print(f"{'-'*5}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*6}")
        for r in sorted(runs, key=lambda r: r["n_concurrent"]):
            print(
                f"{r['n_concurrent']:>5}  "
                f"{r['agg_throughput_tok_s']:>7}  "
                f"{str(r['ttft_avg_ms']):>9}  "
                f"{str(r['ttft_p99_ms']):>9}  "
                f"{str(r['tpot_avg_ms']):>9}  "
                f"{r['wall_time_s']:>6}"
            )

        # vLLM metrics table
        print(f"\n{'n':>5}  {'kv_max':>8}  {'kv_avg':>8}  {'run_max':>8}  {'wait_max':>9}  {'swapped_max':>11}")
        print(f"{'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*11}")
        for r in sorted(runs, key=lambda r: r["n_concurrent"]):
            vm = r.get("vllm_metrics", {})
            kv  = vm.get("kv_cache_usage_perc", {})
            run = vm.get("num_requests_running", {})
            wait= vm.get("num_requests_waiting", {})
            swap= vm.get("num_requests_swapped", {})
            print(
                f"{r['n_concurrent']:>5}  "
                f"{str(kv.get('max','—')):>8}  "
                f"{str(kv.get('avg','—')):>8}  "
                f"{str(run.get('max','—')):>8}  "
                f"{str(wait.get('max','—')):>9}  "
                f"{str(swap.get('max','—')):>11}"
            )

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else LOG_FILE
    records = load_logs(path)
    print(f"Loaded {len(records)} records from {path}")
    summarize(records)