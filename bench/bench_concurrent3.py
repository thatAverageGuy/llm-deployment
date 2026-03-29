import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics
from datetime import datetime
import os
import re


SERVER   = "http://localhost:8000"
MODEL    = "qwen"
PROMPT   = "Explain how a transformer's attention mechanism works in 3 sentences."
LOG_FILE = "logs/bench_results.jsonl"


# ── metrics poller ─────────────────────────────────────────────────────────────

METRICS_TO_TRACK = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_swapped",
]

def parse_metrics(raw: str) -> dict:
    """
    Pull scalar gauge values out of Prometheus text format.
    Each line looks like:
        vllm:kv_cache_usage_perc{engine="0",model_name="qwen"} 0.41
    We want the float at the end.
    """
    result = {}
    for metric in METRICS_TO_TRACK:
        # match the metric name followed by optional labels and a value
        pattern = rf'^{re.escape(metric)}\{{[^}}]*\}}\s+([\d.e+\-]+)'
        for line in raw.splitlines():
            m = re.match(pattern, line)
            if m:
                result[metric] = float(m.group(1))
                break
    return result

class MetricsPoller:
    """
    Polls /metrics in a background thread.
    Call .start() before the run, .stop() after.
    .summary() returns min/max/avg for each tracked metric.
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples = {m:[] for m in METRICS_TO_TRACK}
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop_evt.is_set():
            try:
                resp = requests.get(f"{SERVER}/metrics", timeout=1)
                parsed = parse_metrics(resp.text)
                for metric, val in parsed.items():
                    self.samples[metric].append(val)
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._thread.join()

    def summary(self) -> dict:
        out = {}
        for metric, vals in self.samples.items():
            if not vals:
                out[metric] = {"min": None, "max": None, "avg": None, "samples": 0}
                continue
            short = metric.split("vllm:")[-1]
            out[short] = {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "avg": round(statistics.mean(vals), 4),
                "samples": len(vals)
            }
        return out
    
def measure_single(thread_id: int, max_tokens: int = 200):
    payload = {
        "model":     MODEL,
        "messages":  [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream":    True,
    }
    tokens  = []
    ttft    = None
    t_start = time.perf_counter()

    try:
        with requests.post(
            f"{SERVER}/v1/chat/completions", json=payload, stream=True
        ) as resp:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue
                chunk   = json.loads(line[len("data: "):])
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    t_now = time.perf_counter()
                    tokens.append(content)
                    if ttft is None:
                        ttft = t_now - t_start
    except Exception as e:
        return {"thread_id": thread_id, "error": str(e)}

    t_end      = time.perf_counter()
    total_time = t_end - t_start
    n_tokens   = len(tokens)
    tpot       = (total_time - ttft) / (n_tokens - 1) if n_tokens > 1 else 0.0

    return {
        "thread_id":        thread_id,
        "ttft_ms":          round(ttft * 1000, 2) if ttft else None,
        "tpot_ms":          round(tpot * 1000, 2),
        "total_time_s":     round(total_time, 3),
        "tokens_generated": n_tokens,
        "throughput_tok_s": round(n_tokens / total_time, 1),
    }

# ── concurrent run ─────────────────────────────────────────────────────────────

def run_concurrent(n_concurrent: int, max_tokens: int = 200, tag: str = "", trial: int = 0, log: bool = True):
    print(f"\n--- Sending {n_concurrent} concurrent requests ---")

    poller = MetricsPoller(interval=0.5)
    poller.start()

    t_wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = [
            executor.submit(measure_single, i, max_tokens)
            for i in range(n_concurrent)
        ]
        results = [f.result() for f in futures]
    t_wall_end = time.perf_counter()
    wall_time  = t_wall_end - t_wall_start

    poller.stop()
    gpu_metrics = poller.summary()

    errors     = [r for r in results if "error" in r]
    good       = [r for r in results if "error" not in r]
    ttfts      = [r["ttft_ms"] for r in good if r["ttft_ms"]]
    tpots      = [r["tpot_ms"] for r in good]
    total_toks = sum(r["tokens_generated"] for r in good)

    agg_toks_s = round(total_toks / wall_time, 1)
    ttft_avg   = round(statistics.mean(ttfts), 1)   if ttfts else None
    ttft_p50   = round(statistics.median(ttfts), 1) if ttfts else None
    ttft_p99   = round(sorted(ttfts)[int(len(ttfts) * 0.99) - 1], 1) if len(ttfts) >= 2 else (ttfts[0] if ttfts else None)
    tpot_avg   = round(statistics.mean(tpots), 2)   if tpots else None

    print(f"Completed:          {len(good)}/{n_concurrent}")
    print(f"Errors:             {len(errors)}")
    print(f"Wall time:          {round(wall_time, 2)} s")
    print(f"Total tokens:       {total_toks}")
    print(f"Aggregate tok/s:    {agg_toks_s}")
    print(f"TTFT avg/p50/p99:   {ttft_avg} / {ttft_p50} / {ttft_p99} ms")
    print(f"TPOT avg:           {tpot_avg} ms/token")
    print(f"KV cache max:       {gpu_metrics.get('kv_cache_usage_perc', {}).get('max', 'n/a')}")
    print(f"Max reqs running:   {gpu_metrics.get('num_requests_running', {}).get('max', 'n/a')}")
    print(f"Max reqs waiting:   {gpu_metrics.get('num_requests_waiting', {}).get('max', 'n/a')}")

    record = {
        "timestamp":            datetime.utcnow().isoformat(),
        "record_type":          "trial",
        "trial":                trial,
        "tag":                  tag,
        "n_concurrent":         n_concurrent,
        "max_tokens":           max_tokens,
        "prompt":               PROMPT,
        "wall_time_s":          round(wall_time, 3),
        "total_tokens":         total_toks,
        "agg_throughput_tok_s": agg_toks_s,
        "ttft_avg_ms":          ttft_avg,
        "ttft_p50_ms":          ttft_p50,
        "ttft_p99_ms":          ttft_p99,
        "tpot_avg_ms":          tpot_avg,
        "n_errors":             len(errors),
        "vllm_metrics":         gpu_metrics,   # min/max/avg of each metric during run
        "per_request":          good,
    }

    if log:
        os.makedirs("logs", exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    return record

def run_experiment(n_concurrent: int, n_trials: int = 3, max_tokens: int = 200, tag: str = ""):
    """
    Runs n_trials back-to-back for a given concurrency level.
    Logs each trial individually AND an averaged summary record.
    """
    print(f"\n{'='*50}")
    print(f"n={n_concurrent}, {n_trials} trials, tag={tag}")
    print(f"{'='*50}")

    trial_records = []
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...")
        record = run_concurrent(n_concurrent, max_tokens=max_tokens, tag=tag, trial=trial+1)
        trial_records.append(record)
        time.sleep(2)   # let vLLM drain between trials

    # Average the numeric fields across trials
    def avg(key):
        vals = [r[key] for r in trial_records if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def avg_metric(metric_key, stat):
        vals = [r["vllm_metrics"].get(metric_key, {}).get(stat)
                for r in trial_records]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "timestamp":            datetime.utcnow().isoformat(),
        "tag":                  tag,
        "record_type":          "averaged_summary",   # vs "trial" for individual runs
        "n_concurrent":         n_concurrent,
        "n_trials":             n_trials,
        "max_tokens":           max_tokens,
        "wall_time_s":          avg("wall_time_s"),
        "total_tokens":         avg("total_tokens"),
        "agg_throughput_tok_s": avg("agg_throughput_tok_s"),
        "ttft_avg_ms":          avg("ttft_avg_ms"),
        "ttft_p50_ms":          avg("ttft_p50_ms"),
        "ttft_p99_ms":          avg("ttft_p99_ms"),
        "tpot_avg_ms":          avg("tpot_avg_ms"),
        "n_errors":             avg("n_errors"),
        "vllm_metrics": {
            "kv_cache_usage_perc": {
                "max": avg_metric("kv_cache_usage_perc", "max"),
                "avg": avg_metric("kv_cache_usage_perc", "avg"),
            },
            "num_requests_running": {
                "max": avg_metric("num_requests_running", "max"),
            },
            "num_requests_waiting": {
                "max": avg_metric("num_requests_waiting", "max"),
            },
            "num_requests_swapped": {
                "max": avg_metric("num_requests_swapped", "max"),
            },
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"  → avg tok/s: {summary['agg_throughput_tok_s']}  "
          f"TTFT: {summary['ttft_avg_ms']}ms  "
          f"TPOT: {summary['tpot_avg_ms']}ms")

    return summary


if __name__ == "__main__":
    TAG      = "bf16_baseline"
    N_TRIALS = 3

    # Warmup — 3 silent requests, not logged
    print("Warming up...")
    for _ in range(3):
        run_concurrent(1, max_tokens=50, tag="warmup", log=False)
    print("Warmup done.\n")
    time.sleep(3)

    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        run_experiment(n, n_trials=N_TRIALS, max_tokens=200, tag=TAG)
        time.sleep(5)