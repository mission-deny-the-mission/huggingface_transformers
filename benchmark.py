"""
Benchmarking tool for the OpenAI-compatible API.
"""
import time
import json
import statistics
from typing import List, Dict, Any, Optional
import argparse
import requests
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    prompt: str
    response_text: str
    response_time: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    error: Optional[str] = None


class BenchmarkRunner:
    """Runs benchmarks against the API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []
    
    def run_single_request(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Run a single request and measure performance.
        
        Args:
            prompt: The user prompt to send
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_message: Optional system message
        
        Returns:
            BenchmarkResult with performance metrics
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=300,  # 5 minute timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code != 200:
                return BenchmarkResult(
                    prompt=prompt,
                    response_text="",
                    response_time=response_time,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    tokens_per_second=0.0,
                    error=f"HTTP {response.status_code}: {response.text}",
                )
            
            data = response.json()
            choice = data["choices"][0]
            usage = data["usage"]
            
            completion_text = choice["message"]["content"]
            completion_tokens = usage["completion_tokens"]
            prompt_tokens = usage["prompt_tokens"]
            
            tokens_per_second = completion_tokens / response_time if response_time > 0 else 0.0
            
            return BenchmarkResult(
                prompt=prompt,
                response_text=completion_text,
                response_time=response_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=usage["total_tokens"],
                tokens_per_second=tokens_per_second,
            )
            
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return BenchmarkResult(
                prompt=prompt,
                response_text="",
                response_time=response_time,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                tokens_per_second=0.0,
                error=str(e),
            )
    
    def run_benchmark(
        self,
        prompt: str,
        num_runs: int = 5,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple requests and aggregate statistics.
        
        Args:
            prompt: The user prompt to send
            num_runs: Number of requests to make
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_message: Optional system message
        
        Returns:
            Dictionary with aggregated statistics
        """
        print(f"Running benchmark with {num_runs} requests...")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        results = []
        for i in range(num_runs):
            print(f"Request {i+1}/{num_runs}...", end=" ", flush=True)
            result = self.run_single_request(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_message,
            )
            results.append(result)
            
            if result.error:
                print(f"ERROR: {result.error}")
            else:
                print(f"Done ({result.response_time:.2f}s, {result.tokens_per_second:.2f} tok/s)")
        
        self.results.extend(results)
        
        # Calculate statistics
        successful_results = [r for r in results if not r.error]
        
        if not successful_results:
            return {
                "success": False,
                "error": "All requests failed",
                "total_runs": num_runs,
                "failed_runs": num_runs,
            }
        
        response_times = [r.response_time for r in successful_results]
        tokens_per_second = [r.tokens_per_second for r in successful_results]
        completion_tokens = [r.completion_tokens for r in successful_results]
        prompt_tokens = [r.prompt_tokens for r in successful_results]
        
        stats = {
            "success": True,
            "total_runs": num_runs,
            "successful_runs": len(successful_results),
            "failed_runs": num_runs - len(successful_results),
            "prompt": prompt,
            "response_time": {
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "stdev": statistics.stdev(response_times) if len(response_times) > 1 else 0.0,
            },
            "tokens_per_second": {
                "mean": statistics.mean(tokens_per_second),
                "median": statistics.median(tokens_per_second),
                "min": min(tokens_per_second),
                "max": max(tokens_per_second),
                "stdev": statistics.stdev(tokens_per_second) if len(tokens_per_second) > 1 else 0.0,
            },
            "tokens": {
                "prompt_tokens": {
                    "mean": statistics.mean(prompt_tokens),
                    "median": statistics.median(prompt_tokens),
                },
                "completion_tokens": {
                    "mean": statistics.mean(completion_tokens),
                    "median": statistics.median(completion_tokens),
                    "min": min(completion_tokens),
                    "max": max(completion_tokens),
                },
            },
            "sample_response": successful_results[0].response_text[:500] if successful_results else "",
        }
        
        return stats
    
    def print_stats(self, stats: Dict[str, Any]):
        """Print benchmark statistics in a readable format."""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        if not stats.get("success", False):
            print(f"Benchmark failed: {stats.get('error', 'Unknown error')}")
            return
        
        print(f"\nRuns: {stats['successful_runs']}/{stats['total_runs']} successful")
        print(f"\nPrompt: {stats['prompt'][:200]}..." if len(stats['prompt']) > 200 else f"\nPrompt: {stats['prompt']}")
        
        print("\nResponse Time (seconds):")
        rt = stats['response_time']
        print(f"  Mean:   {rt['mean']:.3f}s")
        print(f"  Median: {rt['median']:.3f}s")
        print(f"  Min:    {rt['min']:.3f}s")
        print(f"  Max:    {rt['max']:.3f}s")
        print(f"  StdDev: {rt['stdev']:.3f}s")
        
        print("\nTokens per Second:")
        tps = stats['tokens_per_second']
        print(f"  Mean:   {tps['mean']:.2f} tok/s")
        print(f"  Median: {tps['median']:.2f} tok/s")
        print(f"  Min:    {tps['min']:.2f} tok/s")
        print(f"  Max:    {tps['max']:.2f} tok/s")
        print(f"  StdDev: {tps['stdev']:.2f} tok/s")
        
        print("\nTokens:")
        tokens = stats['tokens']
        print(f"  Prompt tokens (mean):    {tokens['prompt_tokens']['mean']:.0f}")
        print(f"  Completion tokens (mean): {tokens['completion_tokens']['mean']:.0f}")
        print(f"  Completion tokens (min):  {tokens['completion_tokens']['min']:.0f}")
        print(f"  Completion tokens (max):  {tokens['completion_tokens']['max']:.0f}")
        
        if stats.get('sample_response'):
            print("\nSample Response:")
            print(f"  {stats['sample_response']}...")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the OpenAI-compatible API")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short story about a robot learning to paint.",
        help="The prompt to use for benchmarking",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model identifier",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Optional system message",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON results",
    )
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(base_url=args.url)
    stats = runner.run_benchmark(
        prompt=args.prompt,
        num_runs=args.runs,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_message=args.system_message,
    )
    
    runner.print_stats(stats)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

