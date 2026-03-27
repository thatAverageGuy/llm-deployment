import requests
import json
import time

SERVER = "http://localhost:8000"
MODEL = "qwen"

def measure_single(prompt:str, max_tokens:int=200) -> dict:
    """
    Send one streaming request and measure:
        - TTFT: Time to First Token
        - TPOT: Time Per Output Token
        - total_time: wall clock from send to final token
        - tokens_generated: how many tokens came back
    """

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    tokens = []
    ttft = None
    t_start = time.perf_counter()

    with requests.post(f"{SERVER}/v1/chat/completions", json=payload,stream=True) as resp:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")

            if line == "data: [DONE]":
                break

            if not line.startswith("data: "):
                continue

            json_str = line[len("data: "):]
            chunk = json.loads(json_str)
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", "")

            if content:
                t_now = time.perf_counter()
                tokens.append(content)

                if ttft is None:
                    ttft = t_now - t_start

    t_end = time.perf_counter()
    total_time = t_end - t_start
    n_tokens = len(tokens)

    if n_tokens > 1:
        decode_time = total_time - ttft
        tpot = decode_time / (n_tokens - 1)
    else:
        tpot = 0.0


    return {
        "ttft_ms": round(ttft*1000, 2) if ttft else None,
        "tpot_ms": round(tpot*1000, 2),
        "total_time_s": round(total_time, 3),
        "tokens_generated": n_tokens,
        "throughput_tok_s": round(n_tokens / total_time, 1),
        "response": "".join(tokens)
    }

if __name__=="__main__":
    prompt = "Explain how a transformer's attention mechanism works in 3 sentences."
    
    print("Sending request...")
    print(f"Prompt: {prompt}\n")

    result = measure_single(prompt, max_tokens=200)

    print(f"Response: {result['response']}\n")
    
    print("--- Measurements ---")
    print(f"TTFT: {result['ttft_ms']} ms")
    print(f"TPOT: {result['tpot_ms']} ms/token")
    print(f"Total time: {result['total_time_s']} s")
    print(f"Tokens generated: {result['tokens_generated']}")
    print(f"Throughput: {result['throughput_tok_s']} tok/s")
