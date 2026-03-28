import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor
import statistics


SERVER = "http://localhost:8000"
MODEL = "qwen"

PROMPT = "Explain how a transformer's attention mechanism works in 3 sentences."

def measure_single(thread_id:int, max_tokens:int = 200):
    """
    Same logic as bench_single.py.
    Append result dict into shared results list.
    threading.Lock not needed here - list.append() is atomic in CPython.
    """

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    tokens = []
    ttft = None
    t_start = time.perf_counter()

    try:
        with requests.post(f"{SERVER}/v1/chat/completions", json=payload, stream=True) as resp:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8")
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue

                chunk = json.loads(line[len("data: "):])
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    t_now = time.perf_counter()
                    tokens.append(content)
                    if ttft is None:
                        ttft = t_now - t_start
                    
    except Exception as e:
        return {"thread_id": thread_id, "error": str(e)}
    

    t_end = time.perf_counter()
    total_time = t_end - t_start
    n_tokens = len(tokens)
    tpot = (total_time - ttft) / (n_tokens - 1) if n_tokens > 1 else 0.0

    return {
        "thread_id": thread_id,
        "ttft_ms": round(ttft*1000, 2) if ttft else None,
        "tpot_ms": round(tpot*1000, 2),
        "total_time_s": round(total_time, 3), 
        "tokens_generated": n_tokens,
        "throughput_tok_s": round(n_tokens / total_time, 1)
    }


def run_concurrent(n_concurrent: int):
    results = []

    print(f"\n--- Sending {n_concurrent} concurrent requests ---")

    t_wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = [executor.submit(measure_single, i, 200) for i in range(n_concurrent)]
        results = [f.result() for f in futures]

    t_wall_end = time.perf_counter()
    wall_time = t_wall_end - t_wall_start

    # Aggregate stats
    ttfts       = [r["ttft_ms"] for r in results if "ttft_ms" in r and r["ttft_ms"]]
    tpots       = [r["tpot_ms"] for r in results if "tpot_ms" in r]
    total_toks  = sum(r.get("tokens_generated", 0) for r in results)
    errors      = [r for r in results if "error" in r]

    print(f"Completed:        {len(results) - len(errors)}/{n_concurrent} requests")
    print(f"Errors:           {len(errors)}")
    print(f"Wall time:        {round(wall_time, 2)} s")
    print(f"Total tokens:     {total_toks}")
    print(f"Aggregate tok/s:  {round(total_toks / wall_time, 1)}")
    print(f"TTFT   avg/p50/p99: "
          f"{round(statistics.mean(ttfts),1)} / "
          f"{round(statistics.median(ttfts),1)} / "
          f"{round(sorted(ttfts)[int(len(ttfts)*0.99)-1],1)} ms")
    print(f"TPOT   avg:         {round(statistics.mean(tpots),2)} ms/token")

if __name__=="__main__":
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        run_concurrent(n)
        time.sleep(5) # Let vLLM drain between runs