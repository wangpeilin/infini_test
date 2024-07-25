import argparse
import aiohttp
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple, Optional
import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer
from dataclasses import dataclass
from datetime import datetime

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0
    ttft: float = 0
    prompt_len: int = 0
    output_len: int = 0


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float


async def async_request_generate(
        request_func_input: RequestFuncInput,
        backend,
        pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate")
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "prompt": request_func_input.prompt,
            "n": 1,
            "best_of": request_func_input.best_of,
            "use_beam_search": request_func_input.use_beam_search,
            "temperature": 0.0 if request_func_input.use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "ignore_eos": True,
            "stream": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        ttft = 0
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload, verify_ssl=False) as response:
                if response.status == 200:
                    if backend == "infini":
                        async for data in response.content:
                            if ttft == 0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft
                        output.latency = time.perf_counter() - st
                        # When streaming, '\0' is appended to the end of the response.
                        body = data.decode("utf-8")
                    else:  # vllm
                        async for data in response.content.iter_any():
                            if ttft == 0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft
                        output.latency = time.perf_counter() - st
                        # When streaming, '\0' is appended to the end of the response.
                        body = data.decode("utf-8").strip("\0")
                    try:
                        output.generated_text = json.loads(body)["text"][0][
                                                len(request_func_input.prompt):
                                                ]
                        output.output_len = json.loads(body)["out_len"]
                        output.success = True
                    except json.decoder.JSONDecodeError:
                        output.success = False
                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False
        if pbar:
            pbar.update(1)
        return output


async def async_request_completions(
        request_func_input: RequestFuncInput,
        backend,
        pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("/v1/chat/completions")
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                },
            ],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "ignore_eos": True,
        }
        headers = {'Content-Type': 'application/json'}
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        ttft = 0
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                    output.latency = time.perf_counter() - st
                    # When streaming, '\n\n' is appended to the end of the response.
                    # for completions, the "data: " is in the front, which len == 6
                    body = data.decode("utf-8").split("\n\n")[0][6:]
                    try:
                        if body != '[DONE]':
                            output.output_len = json.loads(body)["usage"]["completion_tokens"]
                        output.success = True
                    except json.decoder.JSONDecodeError:
                        print("json failure!", body, flush=True)
                        output.success = False
                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError, aiohttp.client_exceptions.ClientPayloadError):
            print("fail:", payload["messages"][0]["content"], flush=True)
            output.success = False
        if pbar:
            pbar.update(1)
        return output


def sample_long_bench_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    filtered_dataset: List[Tuple[str, int, int]] = []
    with open(dataset_path, 'r') as lb:
        for line in lb:
            data = json.loads(line)
            pred = tokenizer.encode(data["prompt"])
            max_gen = data["max_gen"]
            filtered_dataset.append((data["prompt"], len(pred), max_gen))
    sampled_requests = filtered_dataset[0:num_requests]

    print("sample_long_bench_requests=======", sampled_requests[0])
    return sampled_requests


def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
    dataset = [dataset[i] for i in range(len(dataset))]
    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))
    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))
    sampled_requests = filtered_dataset[0:num_requests]
    return sampled_requests


async def get_request(
        input_requests: List[Tuple[str, int, int]],
        request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request
        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
        input_requests: List[Tuple[str, int, int]],
        outputs: List[RequestFuncOutput],
        dur_s: float,
        tokenizer: PreTrainedTokenizerBase,
) -> BenchmarkMetrics:
    total_output = 0
    total_input = 0
    completed = 0
    per_token_latencies = []
    ttfts = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            total_output += output_len
            total_input += input_requests[i][1]
            if output_len > 1:
                per_token_latencies.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                )
            ttfts.append(outputs[i].ttft)
            completed += 1
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=total_output / dur_s,
        mean_ttft_ms=np.mean(ttfts) * 1000,
        median_ttft_ms=np.median(ttfts) * 1000,
        p90_ttft_ms=np.percentile(ttfts, 90) * 1000,
        p99_ttft_ms=np.percentile(ttfts, 99) * 1000,
        mean_tpot_ms=np.mean(per_token_latencies) * 1000,
        median_tpot_ms=np.median(per_token_latencies) * 1000,
        p90_tpot_ms=np.percentile(per_token_latencies, 90) * 1000,
        p99_tpot_ms=np.percentile(per_token_latencies, 99) * 1000,
    )
    return metrics


async def benchmark(
        api_url: str,
        model_id: str,
        tokenizer: PreTrainedTokenizerBase,
        input_requests: List[Tuple[str, int, int]],
        best_of: int,
        use_beam_search: bool,
        request_rate: float,
        disable_tqdm: bool,
        backend: str,
):
    print(f"Traffic request rate: {request_rate}")

    # warm_up
    tasks_warm_up = []
    prompt, prompt_len, output_len = input_requests[0]
    request_func_input = RequestFuncInput(
        model=model_id,
        prompt=prompt,
        api_url=api_url,
        prompt_len=prompt_len,
        output_len=output_len,
        best_of=best_of,
        use_beam_search=use_beam_search,
    )
    if api_url.endswith("/generate"):
        request_method = async_request_generate
    elif api_url.endswith("/v1/chat/completions"):
        request_method = async_request_completions
    else:
        raise ValueError(f"not support api: {api_url}, we only support /generate or /v1/chat/completions")
    tasks_warm_up.append(
        asyncio.create_task(
            request_method(request_func_input=request_func_input, backend=backend, pbar=None)
        )
    )
    outputs = await asyncio.gather(*tasks_warm_up)
    print("warm up done.")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_method(request_func_input=request_func_input, backend=backend, pbar=pbar)
            )
        )
    outputs = await asyncio.gather(*tasks)
    if not disable_tqdm:
        pbar.close()
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )
    print(f"Successful requests: {metrics.completed}")
    print(f"Benchmark duration: {benchmark_duration:2f} s")
    print(f"Total input tokens: {metrics.total_input}")
    print(f"Total generated tokens: {metrics.total_output}")
    print(f"Request throughput: {metrics.request_throughput:.2f} requests/s")
    print(f"Input token throughput: {metrics.input_throughput:.2f} tokens/s")
    print(f"Output token throughput: {metrics.output_throughput:.2f} tokens/s")
    print(f"Mean TTFT: {metrics.mean_ttft_ms:.2f} ms")
    print(f"Median TTFT: {metrics.median_ttft_ms:.2f} ms")
    print(f"P90 TTFT: {metrics.p90_ttft_ms:.2f} ms")
    print(f"P99 TTFT: {metrics.p99_ttft_ms:.2f} ms")
    print(f"Mean TPOT: {metrics.mean_tpot_ms:.2f} ms")
    print(f"Median TPOT: {metrics.median_tpot_ms:.2f} ms")
    print(f"P90 TPOT: {metrics.p90_tpot_ms:.2f} ms")
    print(f"P99 TPOT: {metrics.p99_tpot_ms:.2f} ms")
    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_inthroughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p90_ttft_ms": metrics.p90_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p90_tpot_ms": metrics.p90_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
    }
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    print(tokenizer_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=args.trust_remote_code)
    if args.dataset_name == "sharegpt":
        input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)
    elif args.dataset_name == "long_bench":
        input_requests = sample_long_bench_requests(args.dataset, args.num_prompts, tokenizer)
    benchmark_result = asyncio.run(
        benchmark(
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            backend=args.backend,
        )
    )
    # Save config and results to json
    if args.save_result:
        result_json = {}
        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["version"] = args.version
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts
        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}
        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="N/A",
        help="Version of the serving backend/engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/generate",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default model tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
             "then all the requests are sent at time 0. "
             "Otherwise, we use Poisson process to synthesize "
             "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "long_bench"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="infini",
        choices=["infini", "vllm"],
        help="Name of backend to benchmark on.",
    )
    args = parser.parse_args()
    main(args)
