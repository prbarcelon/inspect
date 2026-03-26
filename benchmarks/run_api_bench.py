#!/usr/bin/env python3
"""
Run the inspect API on Martian benchmark PRs and collect findings for judging.

Usage:
    python3 benchmarks/run_api_bench.py
    python3 benchmarks/run_api_bench.py --limit 5
    python3 benchmarks/run_api_bench.py --pr 8087
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

API_URL = "https://inspect-api.fly.dev"
API_KEY = os.environ.get("INSPECT_API_KEY", "insp_5e7be8d37f0ca1a6d4ca45b1326ee649")
BENCHMARK_DATA = Path.home() / "code-review-benchmark" / "offline" / "results" / "benchmark_data.json"
OUTPUT_DIR = Path(__file__).parent / "results" / "api_bench"
POLL_INTERVAL = 5
MAX_POLL_TIME = 420  # 7 min per PR (v13 = 9 lenses + validation + rescue)
CONCURRENCY = 3


def extract_pr_info(url: str):
    """Extract owner/repo and PR number from golden URL."""
    parts = url.rstrip("/").split("/")
    if "/pull/" in url:
        owner_repo = parts[3] + "/" + parts[4]
        pr_number = int(parts[-1])
        return owner_repo, pr_number
    return None, None


async def submit_and_poll(client: httpx.AsyncClient, repo: str, pr_number: int, strategy: str = None) -> dict:
    """Submit a review and poll until complete."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    body = {"repo": repo, "pr_number": pr_number}
    if strategy:
        body["strategy"] = strategy

    # Submit
    resp = await client.post(
        f"{API_URL}/v1/review",
        json=body,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    job = resp.json()
    job_id = job["id"]

    # Poll
    start = time.time()
    while time.time() - start < MAX_POLL_TIME:
        await asyncio.sleep(POLL_INTERVAL)
        resp = await client.get(
            f"{API_URL}/v1/review/{job_id}",
            headers=headers,
            timeout=30,
        )
        data = resp.json()

        if data.get("error") == "job not found":
            return {"status": "lost", "error": "job lost (machine restarted)"}

        status = data.get("status", "unknown")
        if status == "complete":
            return data
        elif status == "failed":
            return {"status": "failed", "error": data.get("error", "unknown")}

    return {"status": "timeout", "error": f"timed out after {MAX_POLL_TIME}s"}


async def run_pr(client: httpx.AsyncClient, url: str, repo: str, pr_number: int, idx: int, total: int, strategy: str = None) -> dict:
    """Run one PR through the API with retry."""
    label = f"[{idx}/{total}] {repo.split('/')[-1]}/#{pr_number}"

    for attempt in range(2):
        try:
            print(f"  {label} {'(retry)' if attempt else ''}...", end=" ", flush=True)
            start = time.time()
            result = await submit_and_poll(client, repo, pr_number, strategy=strategy)
            elapsed = time.time() - start

            status = result.get("status", "unknown")
            if status == "complete":
                findings = result.get("result", {}).get("findings", [])
                detector = result.get("result", {}).get("detector_findings", [])
                print(f"OK — {len(findings)} findings, {len(detector)} detector, {elapsed:.0f}s")
                return {
                    "url": url,
                    "repo": repo,
                    "pr_number": pr_number,
                    "status": "ok",
                    "findings": findings,
                    "detector_findings": detector,
                    "timing": result.get("result", {}).get("timing", {}),
                    "elapsed_s": elapsed,
                }
            elif status == "lost" and attempt == 0:
                print("lost, retrying...", end=" ", flush=True)
                await asyncio.sleep(3)
                continue
            else:
                print(f"{status}: {result.get('error', '?')}")
                return {
                    "url": url,
                    "repo": repo,
                    "pr_number": pr_number,
                    "status": status,
                    "error": result.get("error"),
                    "findings": [],
                    "elapsed_s": elapsed,
                }
        except Exception as e:
            print(f"ERROR: {e}")
            if attempt == 0:
                await asyncio.sleep(5)
                continue
            return {
                "url": url,
                "repo": repo,
                "pr_number": pr_number,
                "status": "error",
                "error": str(e),
                "findings": [],
            }

    return {"url": url, "status": "failed", "findings": []}


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Max PRs to run")
    parser.add_argument("--pr", type=int, help="Run specific PR number only")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--strategy", type=str, help="Review strategy: agentic_v2 or default (hybrid_v10)")
    args = parser.parse_args()

    with open(BENCHMARK_DATA) as f:
        benchmark = json.load(f)

    # Build work list
    work = []
    for url, entry in benchmark.items():
        repo, pr_number = extract_pr_info(url)
        if not repo:
            continue
        if args.pr and pr_number != args.pr:
            continue
        # Fork repos are fine - API can fetch from public GitHub repos
        work.append((url, repo, pr_number))

    if args.limit:
        work = work[:args.limit]

    strategy_label = args.strategy or "hybrid_v10 (default)"
    print(f"Running {len(work)} PRs through inspect API (concurrency={args.concurrency}, strategy={strategy_label})")
    print(f"API: {API_URL}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    results_file = OUTPUT_DIR / "api_results.json"
    existing = {}
    if results_file.exists():
        with open(results_file) as f:
            for r in json.load(f):
                if r.get("status") == "ok":
                    existing[r["url"]] = r

    # Filter out already-done PRs
    todo = [(url, repo, pr) for url, repo, pr in work if url not in existing]
    print(f"Already done: {len(existing)}, remaining: {len(todo)}")

    results = list(existing.values())

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(args.concurrency)
        total = len(todo)

        async def bounded(url, repo, pr, idx):
            async with sem:
                return await run_pr(client, url, repo, pr, idx, total, strategy=args.strategy)

        tasks = [bounded(url, repo, pr, i+1) for i, (url, repo, pr) in enumerate(todo)]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            # Save incrementally
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]
    total_findings = sum(len(r.get("findings", [])) for r in ok)

    print(f"\n{'='*60}")
    print(f"API Benchmark Results")
    print(f"{'='*60}")
    print(f"OK: {len(ok)}/{len(results)}")
    print(f"Failed: {len(failed)}")
    print(f"Total findings: {total_findings}")
    if ok:
        avg_time = sum(r.get("elapsed_s", 0) for r in ok) / len(ok)
        print(f"Avg time: {avg_time:.1f}s")

    # Build candidates.json for the judge
    candidates = {}
    for r in ok:
        url = r["url"]
        issues = [f["issue"] for f in r.get("findings", [])]
        candidates[url] = {
            "inspect-api": [
                {"text": issue, "path": None, "line": None, "source": "api"}
                for issue in issues
            ]
        }

    # Fill missing PRs
    for url in benchmark:
        if url not in candidates:
            candidates[url] = {"inspect-api": []}

    cand_file = OUTPUT_DIR / "candidates.json"
    with open(cand_file, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"\nCandidates: {cand_file}")
    print(f"Total candidate issues: {sum(len(v['inspect-api']) for v in candidates.values())}")


if __name__ == "__main__":
    asyncio.run(main())
