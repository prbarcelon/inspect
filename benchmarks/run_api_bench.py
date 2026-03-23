#!/usr/bin/env python3
"""Run Martian bench PRs through the deployed inspect API and score results."""

import json
import time
import requests
import sys

API_BASE = "https://inspect-api.fly.dev"
API_KEY = sys.argv[1] if len(sys.argv) > 1 else ""
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Load PR list from inspect_results.json
with open("/Users/rohansharma/inspect/benchmarks/martian_results/inspect_results.json") as f:
    results = json.load(f)

prs = []
for url in results.keys():
    # https://github.com/owner/repo/pull/123
    parts = url.replace("https://github.com/", "").split("/pull/")
    repo = parts[0]
    pr_number = int(parts[1])
    prs.append({"repo": repo, "pr_number": pr_number, "url": url})

print(f"Submitting {len(prs)} PRs to inspect API...")

# Submit all review jobs
jobs = {}
for i, pr in enumerate(prs):
    resp = requests.post(
        f"{API_BASE}/v1/review",
        headers=HEADERS,
        json={"repo": pr["repo"], "pr_number": pr["pr_number"]},
    )
    data = resp.json()
    job_id = data.get("id", "")
    status = data.get("status", "error")
    jobs[pr["url"]] = {"id": job_id, "pr": pr, "status": status}
    print(f"[{i+1}/{len(prs)}] {pr['repo']}#{pr['pr_number']} -> {job_id[:8]}... ({status})")
    if "error" in data:
        err = data.get("error", "")
        if err:
            print(f"  ERROR: {err}")

print(f"\nAll {len(jobs)} jobs submitted. Polling for results...")

# Poll until all complete (or timeout after 30 min)
timeout = time.time() + 1800
while time.time() < timeout:
    pending = [url for url, j in jobs.items() if j["status"] not in ("complete", "failed")]
    if not pending:
        break

    print(f"\n  {len(pending)} jobs still pending, checking...")
    for url in pending:
        j = jobs[url]
        try:
            resp = requests.get(
                f"{API_BASE}/v1/review/{j['id']}",
                headers=HEADERS,
            )
            data = resp.json()
            j["status"] = data.get("status", "unknown")
            if j["status"] == "complete":
                findings = data.get("result", {}).get("findings", [])
                j["findings"] = findings
                pr = j["pr"]
                print(f"  DONE: {pr['repo']}#{pr['pr_number']} -> {len(findings)} findings")
            elif j["status"] == "failed":
                j["error"] = data.get("error", "unknown")
                pr = j["pr"]
                print(f"  FAIL: {pr['repo']}#{pr['pr_number']} -> {j['error'][:100]}")
        except Exception as e:
            print(f"  Poll error for {url}: {e}")

    if any(jobs[url]["status"] not in ("complete", "failed") for url in pending):
        time.sleep(10)

# Collect results
api_candidates = {}
for url, j in jobs.items():
    findings = j.get("findings", [])
    candidates = []
    for f in findings:
        issue_text = f.get("issue", "") or f.get("description", "") or str(f)
        candidates.append(issue_text)
    api_candidates[url] = candidates

# Save
out_path = "/Users/rohansharma/inspect/benchmarks/martian_results/api_candidates.json"
with open(out_path, "w") as f:
    json.dump(api_candidates, f, indent=2)

# Summary
total_findings = sum(len(v) for v in api_candidates.values())
completed = sum(1 for j in jobs.values() if j["status"] == "complete")
failed = sum(1 for j in jobs.values() if j["status"] == "failed")
print(f"\nResults: {completed} complete, {failed} failed, {total_findings} total findings")
print(f"Saved to {out_path}")
