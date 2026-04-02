# inspect

Entity-level code review for Git. Scores changes by risk and groups them by logical dependency.

## Structure

Cargo workspace at repo root:
- `inspect-core` — analysis engine, LLM integration, risk scoring
- `inspect-cli` — CLI (`inspect diff`, `inspect pr`, `inspect review`, `inspect predict`)
- `inspect-mcp` — MCP server (6 tools)
- `inspect-api` — REST API (Axum, JWT auth, deployed via Docker)

Separate Next.js website in `docs/` (deployed to inspect.ataraxy-labs.com via Vercel "site" project).

## Build & Test

```bash
cargo build --release -p inspect-cli     # binary at target/release/inspect
cargo build --release -p inspect-mcp     # binary at target/release/inspect-mcp
cargo test --workspace                   # 12 tests
```

## Key Paths

- Risk scoring: `crates/inspect-core/src/`
- CLI commands: `crates/inspect-cli/src/commands/`
- MCP tools: `crates/inspect-mcp/src/`
- API routes: `crates/inspect-api/src/`
- Website: `docs/` (Next.js 15, Clerk auth, Supabase, Stripe billing)
- Benchmarks: `benchmarks/`

## Website (docs/)

The `docs/` directory is the live website. Vercel project name is "site" (root dir set to `docs/`).
- Auth: Clerk
- DB: Supabase
- Billing: Stripe (raw fetch, not SDK)
- Env vars are on the Vercel "site" project, NOT the "docs" project

## How Review Works

1. sem-core extracts entities from diff
2. Score each entity by risk (change classification, blast radius, dependency depth)
3. Group related entities by dependency graph
4. For LLM review: triage first (cut 100 entities to 10), then point LLM at high-risk set

## MCP Tools

`inspect_triage`, `inspect_entity`, `inspect_group`, `inspect_file`, `inspect_stats`, `inspect_risk_map`

## Conventions

- Depends on `sem-core` (git dependency, pinned commit)
- Triage, not review: find where bugs would hurt most, not find bugs
- inspect + LLM > pure LLM (92% token reduction via triage)
- Release on tag push (`v*`)
- License: FSL-1.1-ALv2 (converts to Apache 2.0 after 2 years)
