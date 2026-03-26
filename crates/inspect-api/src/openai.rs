use inspect_core::types::EntityReview;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::prompts;
use crate::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub issue: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    content: Option<String>,
}

#[derive(Deserialize)]
struct IssuesResponse {
    #[serde(default)]
    issues: Vec<serde_json::Value>,
}

// --- Agentic review types ---

pub struct AgentContext {
    pub entity_reviews: Vec<EntityReview>,
    pub repo: String,
    pub base_sha: String,
    pub head_sha: String,
    pub pr_title: String,
    pub diff: String,
    pub triage_section: String,
}

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    id: String,
    output: Vec<ResponseOutput>,
    #[allow(dead_code)]
    status: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponseOutput {
    #[serde(rename = "message")]
    Message { content: Vec<ContentPart> },
    #[serde(rename = "function_call")]
    FunctionCall {
        name: String,
        arguments: String,
        call_id: String,
    },
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[allow(dead_code)]
    text: String,
}

enum AgentStep {
    ToolCalls(Vec<(String, String, String)>), // (name, arguments, call_id)
    Finished(Vec<Finding>),
    Evidence(Vec<EvidenceReport>),
    KeptIndices(Vec<usize>),
    Empty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvidenceReport {
    finding_index: usize,
    finding_issue: String,
    evidence_summary: String,
    code_snippets: Vec<String>,
    verdict: String, // "rescue" or "reject"
}

/// Call OpenAI chat completions API.
async fn call_openai(
    state: &AppState,
    system: &str,
    prompt: &str,
    temperature: f64,
    seed: Option<u64>,
) -> Result<String, String> {
    let mut body = serde_json::json!({
        "model": state.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    });
    if let Some(s) = seed {
        body["seed"] = serde_json::json!(s);
    }

    let resp = state
        .http
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", state.openai_api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("OpenAI API error {status}: {text}"));
    }

    let chat: ChatResponse = resp.json().await.map_err(|e| format!("parse failed: {e}"))?;
    let content = chat
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok(content)
}

async fn call_anthropic(
    state: &AppState,
    system: &str,
    prompt: &str,
    temperature: f64,
) -> Result<String, String> {
    let api_key = state.anthropic_api_key.as_ref()
        .ok_or_else(|| "ANTHROPIC_API_KEY not set".to_string())?;

    let body = serde_json::json!({
        "model": state.anthropic_model,
        "max_tokens": 4096,
        "system": system,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    });

    let resp = state
        .http
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Anthropic request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Anthropic API error {status}: {text}"));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| format!("parse failed: {e}"))?;
    let content = json["content"]
        .as_array()
        .and_then(|arr| {
            arr.iter().find_map(|block| {
                if block["type"].as_str() == Some("text") {
                    block["text"].as_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
        })
        .unwrap_or_default();

    Ok(content)
}

/// Call OpenAI with a specific model override.
async fn call_openai_model(
    state: &AppState,
    model: &str,
    system: &str,
    prompt: &str,
    temperature: f64,
    seed: Option<u64>,
) -> Result<String, String> {
    let mut body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    });
    if let Some(s) = seed {
        body["seed"] = serde_json::json!(s);
    }

    let resp = state
        .http
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", state.openai_api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("OpenAI API error {status}: {text}"));
    }

    let chat: ChatResponse = resp.json().await.map_err(|e| format!("parse failed: {e}"))?;
    let content = chat
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok(content)
}

/// Call Anthropic with a specific model override.
async fn call_anthropic_model(
    state: &AppState,
    model: &str,
    system: &str,
    prompt: &str,
    _temperature: f64,
) -> Result<String, String> {
    let api_key = state.anthropic_api_key.as_ref()
        .ok_or_else(|| "ANTHROPIC_API_KEY not set".to_string())?;

    // Sonnet 4.6 and other thinking models require temperature=1.0
    // Use thinking budget to keep responses focused
    let body = serde_json::json!({
        "model": model,
        "max_tokens": 16000,
        "system": system,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 1.0,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 8000
        }
    });

    let resp = state
        .http
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Anthropic request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Anthropic API error {status}: {text}"));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| format!("parse failed: {e}"))?;
    let content = json["content"]
        .as_array()
        .and_then(|arr| {
            // Find first text block (skip thinking blocks from extended thinking models)
            arr.iter().find_map(|block| {
                if block["type"].as_str() == Some("text") {
                    block["text"].as_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
        })
        .unwrap_or_default();

    if content.is_empty() {
        info!("Anthropic model response had no text block. Raw stop_reason={}, content_types={:?}",
            json["stop_reason"].as_str().unwrap_or("?"),
            json["content"].as_array().map(|arr| arr.iter().map(|b| b["type"].as_str().unwrap_or("?").to_string()).collect::<Vec<_>>()).unwrap_or_default()
        );
    }

    Ok(content)
}

/// Strip markdown code fences and parse JSON issues.
fn parse_issues(text: &str) -> Vec<Finding> {
    let cleaned = strip_code_fences(text);

    // Try direct parse first
    if let Ok(resp) = serde_json::from_str::<IssuesResponse>(&cleaned) {
        return extract_findings(resp);
    }

    // Fallback: find JSON object anywhere in the text (handles prose before/after JSON)
    if let Some(start) = cleaned.find('{') {
        // Find matching closing brace
        let mut depth = 0i32;
        let mut end = start;
        for (i, ch) in cleaned[start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 {
            let json_str = &cleaned[start..end];
            if let Ok(resp) = serde_json::from_str::<IssuesResponse>(json_str) {
                info!("Parsed JSON from embedded response (offset {})", start);
                return extract_findings(resp);
            }
        }
    }

    warn!("Failed to parse LLM response as JSON, text starts with: {}", &cleaned[..cleaned.len().min(100)]);
    Vec::new()
}

fn extract_findings(resp: IssuesResponse) -> Vec<Finding> {
    resp.issues
        .into_iter()
        .filter_map(|v| match v {
            serde_json::Value::String(s) => Some(Finding {
                issue: s,
                evidence: None,
                severity: None,
                file: None,
            }),
            serde_json::Value::Object(map) => {
                let issue = map
                    .get("issue")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if issue.is_empty() {
                    return None;
                }
                Some(Finding {
                    issue,
                    evidence: map.get("evidence").and_then(|v| v.as_str()).map(String::from),
                    severity: map.get("severity").and_then(|v| v.as_str()).map(String::from),
                    file: map.get("file").and_then(|v| v.as_str()).map(String::from),
                })
            }
            _ => None,
        })
        .collect()
}

fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with("```") {
        let after_fence = &trimmed[3..];
        // Skip optional language tag
        let content = if after_fence.starts_with("json") {
            &after_fence[4..]
        } else {
            after_fence
        };
        // Find closing fence
        if let Some(end) = content.rfind("```") {
            return content[..end].trim().to_string();
        }
        return content.trim().to_string();
    }
    trimmed.to_string()
}

/// Extract file paths and basenames from a unified diff.
fn extract_diff_files(diff: &str) -> std::collections::HashSet<String> {
    let mut files = std::collections::HashSet::new();
    for line in diff.lines() {
        if line.starts_with("+++ b/") || line.starts_with("--- a/") {
            let path = &line[6..];
            if path != "/dev/null" && !path.is_empty() {
                files.insert(path.to_string());
                if let Some(basename) = path.rsplit('/').next() {
                    files.insert(basename.to_string());
                }
            }
        }
    }
    files
}

const CODE_EXTENSIONS: &[&str] = &[
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".cs", ".swift", ".kt", ".scala", ".hbs", ".erb",
    ".ex", ".exs", ".hcl",
];

/// Drop findings that reference code files not present in the diff.
fn structural_file_filter(
    findings: Vec<Finding>,
    diff_files: &std::collections::HashSet<String>,
) -> Vec<Finding> {
    if diff_files.is_empty() {
        return findings;
    }
    let lower_basenames: std::collections::HashSet<String> = diff_files
        .iter()
        .map(|f| f.rsplit('/').next().unwrap_or(f).to_lowercase())
        .collect();

    findings
        .into_iter()
        .filter(|f| {
            let text = f.issue.to_lowercase();
            for word in text.replace('/', " / ").split_whitespace() {
                if CODE_EXTENSIONS.iter().any(|ext| word.ends_with(ext)) {
                    let base = word.rsplit('/').next().unwrap_or(word);
                    if !lower_basenames.contains(base) {
                        return false;
                    }
                }
            }
            true
        })
        .collect()
}

/// Validation with seed support.
async fn validate_findings_seeded(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    candidates: &[Finding],
    seed: Option<u64>,
) -> Result<Vec<Finding>, String> {
    let candidates_text: String = candidates
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut line = format!("{}. {}", i + 1, f.issue);
            if let Some(ref ev) = f.evidence {
                line.push_str(&format!("\n   Evidence: {ev}"));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = prompts::format_validate_prompt(pr_title, diff, &candidates_text);
    let text = call_openai(state, prompts::SYSTEM_VALIDATE, &prompt, 0.0, seed).await?;
    Ok(parse_issues(&text))
}

/// Validation using GPT-5.4 specifically.
async fn validate_findings_gpt54(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    candidates: &[Finding],
) -> Result<Vec<Finding>, String> {
    let candidates_text: String = candidates
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut line = format!("{}. {}", i + 1, f.issue);
            if let Some(ref ev) = f.evidence {
                line.push_str(&format!("\n   Evidence: {ev}"));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = prompts::format_validate_prompt(pr_title, diff, &candidates_text);
    let text = call_openai_model(state, "gpt-5.4", prompts::SYSTEM_VALIDATE, &prompt, 0.0, Some(42)).await?;
    Ok(parse_issues(&text))
}

/// Validation using Claude Sonnet 4.6 specifically.
async fn validate_findings_sonnet46(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    candidates: &[Finding],
) -> Result<Vec<Finding>, String> {
    let candidates_text: String = candidates
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut line = format!("{}. {}", i + 1, f.issue);
            if let Some(ref ev) = f.evidence {
                line.push_str(&format!("\n   Evidence: {ev}"));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = prompts::format_validate_prompt(pr_title, diff, &candidates_text);
    let text = call_anthropic_model(state, "claude-sonnet-4-6", prompts::SYSTEM_VALIDATE, &prompt, 0.0).await?;
    if text.is_empty() {
        warn!("Sonnet 4.6 validator returned empty text");
    } else {
        info!("Sonnet 4.6 validator raw response (first 200 chars): {}", &text[..text.len().min(200)]);
    }
    Ok(parse_issues(&text))
}

/// Challenge validation: skeptical second pass that tries to disprove each finding.
async fn challenge_findings(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    candidates: &[Finding],
) -> Result<Vec<Finding>, String> {
    let candidates_text: String = candidates
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut line = format!("{}. {}", i + 1, f.issue);
            if let Some(ref ev) = f.evidence {
                line.push_str(&format!("\n   Evidence: {ev}"));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = prompts::format_challenge_prompt(pr_title, diff, &candidates_text);
    let text = call_openai(state, prompts::SYSTEM_CHALLENGE, &prompt, 0.0, Some(42)).await?;
    Ok(parse_issues(&text))
}

/// Hybrid v19: v10 pipeline + second "challenger" pass.
/// 9 lenses → structural filter → validate → challenge survivors → cap.
pub async fn review_hybrid_v19(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
) -> Vec<Finding> {
    // Run normal v10 pipeline first
    let v10_results = review_hybrid_inner(state, pr_title, diff, triage_section, max_findings, None).await;

    if v10_results.len() <= 2 {
        return v10_results;
    }

    let truncated = prompts::truncate_diff(diff, 65_000);
    let pre_challenge = v10_results.len();

    match challenge_findings(state, pr_title, &truncated, &v10_results).await {
        Ok(challenged) => {
            info!(
                "FUNNEL v19 challenge: {} -> {} ({} dropped)",
                pre_challenge, challenged.len(), pre_challenge - challenged.len()
            );
            for f in &challenged {
                info!("FUNNEL survived challenge: {}", &f.issue[..f.issue.len().min(100)]);
            }
            challenged.into_iter().take(max_findings).collect()
        }
        Err(e) => {
            warn!("Challenge pass failed: {e}, keeping v10 results");
            v10_results
        }
    }
}

/// Raw lenses: 9 parallel lenses + dedup + structural filter, NO validation.
/// Used to measure the lens ceiling (what the lenses generate before filtering).
pub async fn review_raw_lenses(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
) -> Vec<Finding> {
    let truncated = prompts::truncate_diff(diff, 65_000);
    let diff_files = extract_diff_files(diff);

    let p_data = prompts::format_lens_prompt(prompts::PROMPT_LENS_DATA, pr_title, triage_section, &truncated);
    let p_conc = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONCURRENCY, pr_title, triage_section, &truncated);
    let p_cont = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONTRACTS, pr_title, triage_section, &truncated);
    let p_sec = prompts::format_lens_prompt(prompts::PROMPT_LENS_SECURITY, pr_title, triage_section, &truncated);
    let p_typo = prompts::format_lens_prompt(prompts::PROMPT_LENS_TYPOS, pr_title, triage_section, &truncated);
    let p_rt = prompts::format_lens_prompt(prompts::PROMPT_LENS_RUNTIME, pr_title, triage_section, &truncated);
    let p_gen = prompts::format_deep_prompt(pr_title, triage_section, &truncated);

    let (r1, r2, r3, r4, r5, r6, r7, r8, r9) = tokio::join!(
        call_openai(state, prompts::SYSTEM_DATA, &p_data, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_CONCURRENCY, &p_conc, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_CONTRACTS, &p_cont, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_SECURITY, &p_sec, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_TYPOS, &p_typo, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_RUNTIME, &p_rt, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.1, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.1, Some(123)),
    );

    let mut all_findings: Vec<Finding> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for result in [r1, r2, r3, r4, r5, r6, r7, r8, r9] {
        if let Ok(text) = result {
            for f in parse_issues(&text) {
                let key: String = f.issue.to_lowercase().chars().take(80).collect();
                if seen.insert(key) {
                    all_findings.push(f);
                }
            }
        }
    }

    let pre_filter = all_findings.len();
    all_findings = structural_file_filter(all_findings, &diff_files);
    info!(
        "RAW LENSES: {} after dedup, {} after structural filter (no validation)",
        pre_filter, all_findings.len()
    );

    all_findings.into_iter().take(max_findings).collect()
}

/// Hybrid v10 strategy: 9 parallel lenses + structural filter + validation.
pub async fn review_hybrid_v10(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
) -> Vec<Finding> {
    review_hybrid_inner(state, pr_title, diff, triage_section, max_findings, None).await
}

/// Hybrid v12 strategy: 9 parallel lenses + structural filter + AGENTIC validation.
/// Instead of a single blind LLM call for validation, uses an agent with tools
/// (read_file, search_code, get_dependents, etc.) to verify each candidate finding.
pub async fn review_hybrid_v12(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    let candidates = run_9_lenses(state, pr_title, diff, triage_section).await;

    if candidates.is_empty() {
        return (Vec::new(), 0, 0);
    }

    // Skip agentic validation for very few findings
    if candidates.len() <= 2 {
        info!("FUNNEL skip agentic validation (<= 2 findings), returning {} as-is", candidates.len());
        return (candidates, 0, 0);
    }

    // Agentic validation: agent verifies each candidate using tools
    let pre_validation = candidates.len();
    let (validated, iters, calls) = validate_agentic(state, ctx, &candidates, max_findings).await;
    let final_count = validated.len().min(max_findings);
    info!(
        "FUNNEL agentic validation: {} -> {} ({} dropped), {} iters, {} tool calls",
        pre_validation, validated.len(), pre_validation.saturating_sub(validated.len()), iters, calls
    );
    (validated.into_iter().take(max_findings).collect(), iters, calls)
}

/// Hybrid v13 strategy: 9 lenses + blind validation + agent rescue pass.
/// First runs blind LLM validation (good at killing FPs), then lets an agent
/// investigate the KILLED findings to rescue ones that need cross-file evidence.
pub async fn review_hybrid_v13(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    let candidates = run_9_lenses(state, pr_title, diff, triage_section).await;

    if candidates.is_empty() {
        return (Vec::new(), 0, 0);
    }

    if candidates.len() <= 2 {
        info!("FUNNEL skip validation (<= 2 findings), returning {} as-is", candidates.len());
        return (candidates, 0, 0);
    }

    // Step 1: Blind validation (same as v10)
    let truncated = prompts::truncate_diff(diff, 65_000);
    let pre_validation = candidates.len();

    let kept = match validate_findings_seeded(state, pr_title, &truncated, &candidates, Some(42)).await {
        Ok(v) => v,
        Err(e) => {
            warn!("Blind validation failed: {e}, returning candidates as-is");
            return (candidates.into_iter().take(max_findings).collect(), 0, 0);
        }
    };

    // Figure out which findings were killed
    let kept_keys: std::collections::HashSet<String> = kept
        .iter()
        .map(|f| f.issue.to_lowercase().chars().take(80).collect())
        .collect();

    let killed: Vec<Finding> = candidates
        .into_iter()
        .filter(|f| {
            let key: String = f.issue.to_lowercase().chars().take(80).collect();
            !kept_keys.contains(&key)
        })
        .collect();

    info!(
        "FUNNEL blind validation: {} -> {} kept, {} killed",
        pre_validation, kept.len(), killed.len()
    );

    // Step 2: Agent rescue pass on killed findings
    if killed.is_empty() {
        info!("FUNNEL no killed findings to rescue");
        return (kept.into_iter().take(max_findings).collect(), 0, 0);
    }

    let (rescued, iters, calls) = rescue_agentic(state, ctx, &killed).await;
    info!(
        "FUNNEL agent rescue: {} killed -> {} rescued, {} iters, {} calls",
        killed.len(), rescued.len(), iters, calls
    );

    // Combine: kept + rescued, cap at max_findings
    let mut final_findings = kept;
    for f in rescued {
        let key: String = f.issue.to_lowercase().chars().take(80).collect();
        if !kept_keys.contains(&key) {
            final_findings.push(f);
        }
    }

    let total = final_findings.len().min(max_findings);
    info!("FUNNEL final: {} findings (capped to {})", final_findings.len(), total);
    (final_findings.into_iter().take(max_findings).collect(), iters, calls)
}

/// Hybrid v14 strategy: 9 lenses + blind validation + agent evidence gathering + revalidation.
/// Agent investigates killed findings and gathers evidence, but the blind validator
/// makes the final keep/drop decision with the evidence attached.
pub async fn review_hybrid_v14(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    let candidates = run_9_lenses(state, pr_title, diff, triage_section).await;

    if candidates.is_empty() {
        return (Vec::new(), 0, 0);
    }

    if candidates.len() <= 2 {
        info!("FUNNEL skip validation (<= 2 findings), returning {} as-is", candidates.len());
        return (candidates, 0, 0);
    }

    // Step 1: Blind validation (same as v10)
    let truncated = prompts::truncate_diff(diff, 65_000);
    let pre_validation = candidates.len();

    let kept = match validate_findings_seeded(state, pr_title, &truncated, &candidates, Some(42)).await {
        Ok(v) => v,
        Err(e) => {
            warn!("Blind validation failed: {e}, returning candidates as-is");
            return (candidates.into_iter().take(max_findings).collect(), 0, 0);
        }
    };

    // Figure out which findings were killed
    let kept_keys: std::collections::HashSet<String> = kept
        .iter()
        .map(|f| f.issue.to_lowercase().chars().take(80).collect())
        .collect();

    let killed: Vec<Finding> = candidates
        .into_iter()
        .filter(|f| {
            let key: String = f.issue.to_lowercase().chars().take(80).collect();
            !kept_keys.contains(&key)
        })
        .collect();

    info!(
        "FUNNEL blind validation: {} -> {} kept, {} killed",
        pre_validation, kept.len(), killed.len()
    );

    // Step 2: Agent evidence gathering + revalidation on killed findings
    if killed.is_empty() {
        info!("FUNNEL no killed findings to investigate");
        return (kept.into_iter().take(max_findings).collect(), 0, 0);
    }

    let (rescued, iters, calls) = rescue_with_evidence(state, ctx, &killed, pr_title, diff).await;
    info!(
        "FUNNEL evidence rescue: {} killed -> {} rescued after revalidation, {} iters, {} calls",
        killed.len(), rescued.len(), iters, calls
    );

    // Combine: kept + rescued, cap at max_findings
    let mut final_findings = kept;
    for f in rescued {
        let key: String = f.issue.to_lowercase().chars().take(80).collect();
        if !kept_keys.contains(&key) {
            final_findings.push(f);
        }
    }

    let total = final_findings.len().min(max_findings);
    info!("FUNNEL final: {} findings (capped to {})", final_findings.len(), total);
    (final_findings.into_iter().take(max_findings).collect(), iters, calls)
}

/// Hybrid v11 strategy: 9 parallel lenses + agentic 10th lens + structural filter + validation.
pub async fn review_hybrid_v11(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    let findings = review_hybrid_inner(state, pr_title, diff, triage_section, max_findings, Some(ctx)).await;
    // Agent stats are logged inside review_hybrid_inner; return placeholder counts
    // (the real counts come from the agent call inside)
    (findings, 0, 0)
}

/// Shared implementation for hybrid v10/v11.
async fn review_hybrid_inner(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    agent_ctx: Option<&AgentContext>,
) -> Vec<Finding> {
    let truncated = prompts::truncate_diff(diff, 65_000);
    let diff_files = extract_diff_files(diff);

    // Build 6 specialized lens prompts
    let p_data = prompts::format_lens_prompt(prompts::PROMPT_LENS_DATA, pr_title, triage_section, &truncated);
    let p_conc = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONCURRENCY, pr_title, triage_section, &truncated);
    let p_cont = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONTRACTS, pr_title, triage_section, &truncated);
    let p_sec = prompts::format_lens_prompt(prompts::PROMPT_LENS_SECURITY, pr_title, triage_section, &truncated);
    let p_typo = prompts::format_lens_prompt(prompts::PROMPT_LENS_TYPOS, pr_title, triage_section, &truncated);
    let p_rt = prompts::format_lens_prompt(prompts::PROMPT_LENS_RUNTIME, pr_title, triage_section, &truncated);

    // 3 general lens prompts
    let p_gen = prompts::format_deep_prompt(pr_title, triage_section, &truncated);

    // 9 lenses + optional agent in parallel
    let lens_future = async {
        tokio::join!(
            call_openai(state, prompts::SYSTEM_DATA, &p_data, 0.0, Some(42)),
            call_openai(state, prompts::SYSTEM_CONCURRENCY, &p_conc, 0.0, Some(42)),
            call_openai(state, prompts::SYSTEM_CONTRACTS, &p_cont, 0.0, Some(42)),
            call_openai(state, prompts::SYSTEM_SECURITY, &p_sec, 0.0, Some(42)),
            call_openai(state, prompts::SYSTEM_TYPOS, &p_typo, 0.0, Some(42)),
            call_openai(state, prompts::SYSTEM_RUNTIME, &p_rt, 0.0, Some(42)),
            call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.0, Some(42)),
            call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.1, Some(42)),
            call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.1, Some(123)),
        )
    };

    let agent_future = async {
        if let Some(ctx) = agent_ctx {
            let (findings, iters, calls) = review_agentic_v2(state, ctx).await;
            info!("Agent 10th lens: {} findings, {} iters, {} calls", findings.len(), iters, calls);
            findings
        } else {
            Vec::new()
        }
    };

    let ((r1, r2, r3, r4, r5, r6, r7, r8, r9), agent_findings) =
        tokio::join!(lens_future, agent_future);

    // Merge + dedup (lenses)
    let mut all_findings: Vec<Finding> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut lens_count = 0usize;

    for result in [r1, r2, r3, r4, r5, r6, r7, r8, r9] {
        if let Ok(text) = result {
            for f in parse_issues(&text) {
                let key: String = f.issue.to_lowercase().chars().take(80).collect();
                if seen.insert(key) {
                    all_findings.push(f);
                    lens_count += 1;
                }
            }
        } else {
            warn!("Lens failed: {:?}", result.err());
        }
    }

    // Merge agent findings (track separately)
    let mut agent_added = 0usize;
    let mut agent_deduped = 0usize;
    for f in &agent_findings {
        let key: String = f.issue.to_lowercase().chars().take(80).collect();
        if seen.insert(key) {
            all_findings.push(f.clone());
            agent_added += 1;
        } else {
            agent_deduped += 1;
        }
    }
    let agent_total = agent_findings.len();

    info!(
        "FUNNEL merge: {} lens + {} agent raw ({} added, {} deduped) = {} total",
        lens_count, agent_total, agent_added, agent_deduped, all_findings.len()
    );

    if all_findings.is_empty() {
        return Vec::new();
    }

    // Structural file filter
    let pre_filter = all_findings.len();
    all_findings = structural_file_filter(all_findings, &diff_files);
    info!(
        "FUNNEL structural filter: {} -> {} ({} dropped)",
        pre_filter, all_findings.len(), pre_filter - all_findings.len()
    );

    if all_findings.is_empty() {
        return Vec::new();
    }

    // Skip validation if few findings
    if all_findings.len() <= 2 {
        info!("FUNNEL skip validation (<= 2 findings), returning {} as-is", all_findings.len());
        return all_findings;
    }

    // Validation pass with seed=42
    let pre_validation = all_findings.len();
    match validate_findings_seeded(state, pr_title, &truncated, &all_findings, Some(42)).await {
        Ok(validated) => {
            let post_validation = validated.len();
            let final_count = post_validation.min(max_findings);
            info!(
                "FUNNEL validation: {} -> {} ({} dropped), top-{} = {}",
                pre_validation, post_validation, pre_validation - post_validation,
                max_findings, final_count
            );
            // Log which findings survived
            for f in &validated {
                info!("FUNNEL survived: {}", &f.issue[..f.issue.len().min(100)]);
            }
            validated.into_iter().take(max_findings).collect()
        }
        Err(e) => {
            warn!("Validation failed: {e}");
            all_findings.into_iter().take(max_findings).collect()
        }
    }
}

/// Run 9 parallel lenses, merge with agreement counting, structural filter.
/// Returns each finding with how many lenses found it.
async fn run_9_lenses_with_counts(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
) -> Vec<(Finding, usize)> {
    let truncated = prompts::truncate_diff(diff, 65_000);
    let diff_files = extract_diff_files(diff);

    let p_data = prompts::format_lens_prompt(prompts::PROMPT_LENS_DATA, pr_title, triage_section, &truncated);
    let p_conc = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONCURRENCY, pr_title, triage_section, &truncated);
    let p_cont = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONTRACTS, pr_title, triage_section, &truncated);
    let p_sec = prompts::format_lens_prompt(prompts::PROMPT_LENS_SECURITY, pr_title, triage_section, &truncated);
    let p_typo = prompts::format_lens_prompt(prompts::PROMPT_LENS_TYPOS, pr_title, triage_section, &truncated);
    let p_rt = prompts::format_lens_prompt(prompts::PROMPT_LENS_RUNTIME, pr_title, triage_section, &truncated);
    let p_gen = prompts::format_deep_prompt(pr_title, triage_section, &truncated);

    let (r1, r2, r3, r4, r5, r6, r7, r8, r9) = tokio::join!(
        call_openai(state, prompts::SYSTEM_DATA, &p_data, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_CONCURRENCY, &p_conc, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_CONTRACTS, &p_cont, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_SECURITY, &p_sec, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_TYPOS, &p_typo, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_RUNTIME, &p_rt, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.1, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.1, Some(123)),
    );

    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut first_seen: std::collections::HashMap<String, Finding> = std::collections::HashMap::new();

    for result in [r1, r2, r3, r4, r5, r6, r7, r8, r9] {
        if let Ok(text) = result {
            let mut lens_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
            for f in parse_issues(&text) {
                let key: String = f.issue.to_lowercase().chars().take(80).collect();
                if lens_keys.insert(key.clone()) {
                    *counts.entry(key.clone()).or_insert(0) += 1;
                    first_seen.entry(key).or_insert(f);
                }
            }
        } else {
            warn!("Lens failed: {:?}", result.err());
        }
    }

    let total_unique = first_seen.len();

    let mut all_findings: Vec<(Finding, usize)> = first_seen
        .into_iter()
        .map(|(key, f)| {
            let count = counts.get(&key).copied().unwrap_or(1);
            (f, count)
        })
        .collect();

    info!("FUNNEL 9-lens merge: {} unique findings", total_unique);

    let mut dist: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for count in counts.values() {
        *dist.entry(*count).or_insert(0) += 1;
    }
    let mut dist_sorted: Vec<_> = dist.into_iter().collect();
    dist_sorted.sort();
    info!("FUNNEL lens agreement distribution: {:?}", dist_sorted);

    if all_findings.is_empty() {
        return Vec::new();
    }

    let pre_filter = all_findings.len();
    let diff_files_ref = &diff_files;
    let findings_only: Vec<Finding> = all_findings.iter().map(|(f, _)| f.clone()).collect();
    let filtered = structural_file_filter(findings_only, diff_files_ref);
    let filtered_keys: std::collections::HashSet<String> = filtered
        .iter()
        .map(|f| f.issue.to_lowercase().chars().take(80).collect())
        .collect();
    all_findings.retain(|(f, _)| {
        let key: String = f.issue.to_lowercase().chars().take(80).collect();
        filtered_keys.contains(&key)
    });
    info!(
        "FUNNEL structural filter: {} -> {} ({} dropped)",
        pre_filter, all_findings.len(), pre_filter - all_findings.len()
    );

    all_findings
}

/// Run 9 parallel lenses, merge, dedup, structural filter. Returns candidates pre-validation.
async fn run_9_lenses(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
) -> Vec<Finding> {
    run_9_lenses_with_counts(state, pr_title, diff, triage_section)
        .await
        .into_iter()
        .map(|(f, _)| f)
        .collect()
}

/// Run 9 lenses but only keep findings that appear in min_agreement or more lenses.
async fn run_9_lenses_intersection(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    min_agreement: usize,
) -> Vec<Finding> {
    let truncated = prompts::truncate_diff(diff, 65_000);
    let diff_files = extract_diff_files(diff);

    let p_data = prompts::format_lens_prompt(prompts::PROMPT_LENS_DATA, pr_title, triage_section, &truncated);
    let p_conc = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONCURRENCY, pr_title, triage_section, &truncated);
    let p_cont = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONTRACTS, pr_title, triage_section, &truncated);
    let p_sec = prompts::format_lens_prompt(prompts::PROMPT_LENS_SECURITY, pr_title, triage_section, &truncated);
    let p_typo = prompts::format_lens_prompt(prompts::PROMPT_LENS_TYPOS, pr_title, triage_section, &truncated);
    let p_rt = prompts::format_lens_prompt(prompts::PROMPT_LENS_RUNTIME, pr_title, triage_section, &truncated);
    let p_gen = prompts::format_deep_prompt(pr_title, triage_section, &truncated);

    let (r1, r2, r3, r4, r5, r6, r7, r8, r9) = tokio::join!(
        call_openai(state, prompts::SYSTEM_DATA, &p_data, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_CONCURRENCY, &p_conc, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_CONTRACTS, &p_cont, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_SECURITY, &p_sec, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_TYPOS, &p_typo, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_RUNTIME, &p_rt, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.1, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.1, Some(123)),
    );

    // Count how many lenses each finding appears in
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut first_seen: std::collections::HashMap<String, Finding> = std::collections::HashMap::new();

    for result in [r1, r2, r3, r4, r5, r6, r7, r8, r9] {
        if let Ok(text) = result {
            let mut lens_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
            for f in parse_issues(&text) {
                let key: String = f.issue.to_lowercase().chars().take(80).collect();
                if lens_keys.insert(key.clone()) {
                    *counts.entry(key.clone()).or_insert(0) += 1;
                    first_seen.entry(key).or_insert(f);
                }
            }
        } else {
            warn!("Lens failed: {:?}", result.err());
        }
    }

    let total_unique = first_seen.len();

    let mut all_findings: Vec<Finding> = first_seen
        .into_iter()
        .filter(|(key, _)| counts.get(key).copied().unwrap_or(0) >= min_agreement)
        .map(|(_, f)| f)
        .collect();

    info!(
        "FUNNEL 9-lens intersection (>={} lenses): {} unique -> {} agreed",
        min_agreement, total_unique, all_findings.len()
    );

    let mut dist: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for count in counts.values() {
        *dist.entry(*count).or_insert(0) += 1;
    }
    let mut dist_sorted: Vec<_> = dist.into_iter().collect();
    dist_sorted.sort();
    info!("FUNNEL lens agreement distribution: {:?}", dist_sorted);

    if all_findings.is_empty() {
        return Vec::new();
    }

    let pre_filter = all_findings.len();
    all_findings = structural_file_filter(all_findings, &diff_files);
    info!(
        "FUNNEL structural filter: {} -> {} ({} dropped)",
        pre_filter, all_findings.len(), pre_filter - all_findings.len()
    );

    all_findings
}

/// Hybrid v16: 9 lenses + enrichment (agreement + triage metadata) + tuned blind validation.
/// Enriches each finding with lens agreement counts and entity risk/dependents,
/// then a single blind validator with explicit decision rules based on the metadata.
/// No agentic pass. The prompt does the heavy lifting.
pub async fn review_hybrid_v16(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    entity_reviews: &[EntityReview],
) -> Vec<Finding> {
    let candidates = run_9_lenses_with_counts(state, pr_title, diff, triage_section).await;

    if candidates.is_empty() {
        return Vec::new();
    }

    if candidates.len() <= 2 {
        info!("FUNNEL skip validation (<= 2 findings), returning {} as-is", candidates.len());
        return candidates.into_iter().map(|(f, _)| f).collect();
    }

    // Enrich each finding with agreement count + entity metadata
    let enriched: Vec<Finding> = candidates
        .iter()
        .map(|(f, lens_count)| {
            let mut enriched_issue = f.issue.clone();
            enriched_issue.push_str(&format!("\n   [Flagged by {}/9 independent reviewers]", lens_count));

            // Match finding to entity reviews by file path
            let matched_entity = if let Some(ref file_path) = f.file {
                entity_reviews.iter().find(|e| {
                    e.file_path.contains(file_path) || file_path.contains(&e.file_path)
                })
            } else {
                None
            };

            // Fallback: match by entity name mentioned in issue text
            let matched_entity = matched_entity.or_else(|| {
                let issue_lower = f.issue.to_lowercase();
                entity_reviews.iter().find(|e| {
                    let name_lower = e.entity_name.to_lowercase();
                    let short_name = name_lower.rsplit("::").next().unwrap_or(&name_lower);
                    issue_lower.contains(short_name) && short_name.len() >= 3
                })
            });

            if let Some(entity) = matched_entity {
                let visibility = if entity.is_public_api { "public" } else { "internal" };
                enriched_issue.push_str(&format!(
                    "\n   [Entity: {} | risk: {:.2} | {} dependents | {}]",
                    entity.entity_name, entity.risk_score, entity.dependent_count, visibility
                ));
            }

            Finding {
                issue: enriched_issue,
                evidence: f.evidence.clone(),
                severity: f.severity.clone(),
                file: f.file.clone(),
            }
        })
        .collect();

    info!("FUNNEL enriched {} findings with agreement + triage metadata", enriched.len());

    let truncated = prompts::truncate_diff(diff, 65_000);
    let pre_validation = enriched.len();

    let validated = match validate_findings_seeded(state, pr_title, &truncated, &enriched, Some(42)).await {
        Ok(v) => v,
        Err(e) => {
            warn!("Validation failed: {e}, returning candidates as-is");
            return candidates.into_iter().take(max_findings).map(|(f, _)| f).collect();
        }
    };

    // Strip metadata lines from survivors
    let cleaned: Vec<Finding> = validated
        .into_iter()
        .map(|f| {
            let cleaned_issue = f.issue
                .lines()
                .filter(|line| {
                    let trimmed = line.trim();
                    !trimmed.starts_with("[Flagged by ") && !trimmed.starts_with("[Entity: ")
                })
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();
            Finding {
                issue: cleaned_issue,
                evidence: f.evidence,
                severity: f.severity,
                file: f.file,
            }
        })
        .collect();

    info!(
        "FUNNEL validation (enriched): {} -> {} ({} dropped)",
        pre_validation, cleaned.len(), pre_validation - cleaned.len()
    );

    for f in &cleaned {
        info!("FUNNEL survived: {}", &f.issue[..f.issue.len().min(100)]);
    }

    cleaned.into_iter().take(max_findings).collect()
}

/// Hybrid v17: 2+/9 agreement = auto-keep, 1/9 = validate normally.
/// Findings with multi-lens agreement bypass validation (protected).
/// Single-lens findings still go through the standard v10 validation.
pub async fn review_hybrid_v17(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
) -> Vec<Finding> {
    let candidates = run_9_lenses_with_counts(state, pr_title, diff, triage_section).await;

    if candidates.is_empty() {
        return Vec::new();
    }

    // Split by agreement
    let mut auto_keep: Vec<Finding> = Vec::new();
    let mut to_validate: Vec<Finding> = Vec::new();

    for (f, count) in &candidates {
        if *count >= 2 {
            auto_keep.push(f.clone());
        } else {
            to_validate.push(f.clone());
        }
    }

    info!(
        "FUNNEL v17 split: {} auto-keep (2+/9), {} to validate (1/9)",
        auto_keep.len(), to_validate.len()
    );

    // Validate the 1/9 tier (same as v10)
    let mut final_findings = auto_keep;

    if to_validate.len() > 2 {
        let truncated = prompts::truncate_diff(diff, 65_000);
        match validate_findings_seeded(state, pr_title, &truncated, &to_validate, Some(42)).await {
            Ok(validated) => {
                info!(
                    "FUNNEL v17 validation (1/9 tier): {} -> {} ({} dropped)",
                    to_validate.len(), validated.len(), to_validate.len() - validated.len()
                );
                final_findings.extend(validated);
            }
            Err(e) => {
                warn!("Validation failed: {e}, keeping all 1/9 findings");
                final_findings.extend(to_validate);
            }
        }
    } else {
        info!("FUNNEL v17 skip validation for 1/9 tier (<= 2 findings)");
        final_findings.extend(to_validate);
    }

    for f in &final_findings {
        info!("FUNNEL survived: {}", &f.issue[..f.issue.len().min(100)]);
    }

    final_findings.into_iter().take(max_findings).collect()
}

/// Validate findings using Claude (Anthropic) as a second opinion.
async fn validate_findings_anthropic(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    candidates: &[Finding],
) -> Result<Vec<Finding>, String> {
    let candidates_text: String = candidates
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut line = format!("{}. {}", i + 1, f.issue);
            if let Some(ref ev) = f.evidence {
                line.push_str(&format!("\n   Evidence: {ev}"));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = prompts::format_validate_prompt(pr_title, diff, &candidates_text);
    let text = call_anthropic(state, prompts::SYSTEM_VALIDATE, &prompt, 0.0).await?;
    Ok(parse_issues(&text))
}

/// Hybrid v18: cross-model parallel validation.
/// 9 lenses (GPT) → structural filter → GPT + Claude validate in parallel → intersection.
/// A finding must survive BOTH validators to be kept.
pub async fn review_hybrid_v18(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
) -> Vec<Finding> {
    let candidates = run_9_lenses(state, pr_title, diff, triage_section).await;

    if candidates.is_empty() {
        return Vec::new();
    }

    if candidates.len() <= 2 {
        info!("FUNNEL v18 skip validation (<= 2 findings), returning {} as-is", candidates.len());
        return candidates;
    }

    let truncated = prompts::truncate_diff(diff, 65_000);
    let pre_validation = candidates.len();

    // Run GPT and Claude validation in parallel on the same candidates
    let (gpt_result, claude_result) = tokio::join!(
        validate_findings_seeded(state, pr_title, &truncated, &candidates, Some(42)),
        validate_findings_anthropic(state, pr_title, &truncated, &candidates)
    );

    let gpt_validated = match gpt_result {
        Ok(v) => v,
        Err(e) => {
            warn!("GPT validation failed: {e}, using all candidates for GPT set");
            candidates.clone()
        }
    };

    let claude_validated = match claude_result {
        Ok(v) => v,
        Err(e) => {
            warn!("Claude validation failed: {e}, using all candidates for Claude set");
            candidates.clone()
        }
    };

    info!(
        "FUNNEL v18 parallel: {} candidates -> GPT kept {}, Claude kept {}",
        pre_validation, gpt_validated.len(), claude_validated.len()
    );

    // Intersection: keep findings that BOTH models approved
    // Match by first 80 chars lowercase (same dedup key)
    let claude_keys: std::collections::HashSet<String> = claude_validated
        .iter()
        .map(|f| f.issue.to_lowercase().chars().take(80).collect())
        .collect();

    let mut final_findings: Vec<Finding> = gpt_validated
        .into_iter()
        .filter(|f| {
            let key: String = f.issue.to_lowercase().chars().take(80).collect();
            claude_keys.contains(&key)
        })
        .collect();

    info!(
        "FUNNEL v18 intersection: {} findings survived both models",
        final_findings.len()
    );

    // If intersection is empty but both had results, fall back to GPT (don't lose everything)
    if final_findings.is_empty() && !candidates.is_empty() {
        warn!("FUNNEL v18 intersection empty, falling back to GPT validation");
        match validate_findings_seeded(state, pr_title, &truncated, &candidates, Some(42)).await {
            Ok(v) => final_findings = v,
            Err(_) => return candidates.into_iter().take(max_findings).collect(),
        }
    }

    for f in &final_findings {
        info!("FUNNEL survived: {}", &f.issue[..f.issue.len().min(100)]);
    }

    final_findings.into_iter().take(max_findings).collect()
}

/// Hybrid v15: 9 lenses with intersection (2+ agreement) + blind validation. No rescue.
pub async fn review_hybrid_v15(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
) -> Vec<Finding> {
    let candidates = run_9_lenses_intersection(state, pr_title, diff, triage_section, 2).await;

    if candidates.is_empty() {
        return Vec::new();
    }

    if candidates.len() <= 2 {
        info!("FUNNEL skip validation (<= 2 findings), returning {} as-is", candidates.len());
        return candidates;
    }

    let truncated = prompts::truncate_diff(diff, 65_000);
    let pre_validation = candidates.len();

    let validated = match validate_findings_seeded(state, pr_title, &truncated, &candidates, Some(42)).await {
        Ok(v) => v,
        Err(e) => {
            warn!("Validation failed: {e}, returning candidates as-is");
            return candidates.into_iter().take(max_findings).collect();
        }
    };

    let final_count = validated.len().min(max_findings);
    info!(
        "FUNNEL validation: {} -> {} (capped to {})",
        pre_validation, validated.len(), final_count
    );

    validated.into_iter().take(max_findings).collect()
}

/// Agentic validation: agent verifies candidate findings using tools.
/// Instead of a single blind LLM call, the agent can read_file, search_code,
/// get_dependents to verify cross-file issues that the blind validator kills.
async fn validate_agentic(
    state: &AppState,
    ctx: &AgentContext,
    candidates: &[Finding],
    max_findings: usize,
) -> (Vec<Finding>, usize, usize) {
    let tools = build_tool_definitions();
    let instructions = prompts::SYSTEM_AGENT_VALIDATE;
    let initial_prompt = prompts::build_agent_validate_prompt(ctx, candidates);

    let mut resp = match call_responses_api(
        state,
        instructions,
        Some(&initial_prompt),
        &tools,
        None,
        None,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("Agentic validation initial call failed: {e}");
            // Fall back to returning top candidates as-is
            return (candidates.iter().take(max_findings).cloned().collect(), 0, 0);
        }
    };

    let max_iterations = 25;
    let mut total_tool_calls = 0usize;
    let mut iterations = 0usize;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            warn!("Agentic validation hit max iterations ({max_iterations})");
            break;
        }

        match parse_agent_step(&resp.output) {
            AgentStep::Finished(findings) => {
                info!(
                    "Agentic validation finished: {} verified findings after {} iterations, {} tool calls",
                    findings.len(), iterations, total_tool_calls
                );
                return (findings, iterations, total_tool_calls);
            }
            AgentStep::ToolCalls(calls) => {
                total_tool_calls += calls.len();
                info!(
                    "Agentic validation iter {}: {} tool calls (total: {})",
                    iterations, calls.len(), total_tool_calls
                );

                let mut tool_outputs = Vec::new();
                for (name, arguments, call_id) in &calls {
                    let result = match name.as_str() {
                        "get_entity" => execute_get_entity(ctx, arguments),
                        "read_file" => execute_read_file(state, ctx, arguments).await,
                        "search_code" => execute_search_code(state, ctx, arguments).await,
                        "get_dependents" => execute_get_dependents(ctx, arguments),
                        "get_dependencies" => execute_get_dependencies(ctx, arguments),
                        _ => format!(r#"{{"error": "unknown tool: {name}"}}"#),
                    };

                    tool_outputs.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }));
                }

                let prev_id = resp.id.clone();
                resp = match call_responses_api(
                    state,
                    instructions,
                    None,
                    &tools,
                    Some(&prev_id),
                    Some(tool_outputs),
                )
                .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        warn!("Agentic validation continuation failed: {e}");
                        break;
                    }
                };
            }
            AgentStep::Evidence(_) => {
                info!("Agentic validation got evidence instead of findings");
                break;
            }
            AgentStep::KeptIndices(_) => {
                info!("Agentic validation got kept_indices unexpectedly");
                break;
            }
            AgentStep::Empty => {
                info!("Agentic validation returned empty");
                break;
            }
        }
    }

    // If agent didn't submit findings, fall back to returning candidates
    warn!("Agentic validation did not submit findings, returning top candidates as fallback");
    (candidates.iter().take(max_findings).cloned().collect(), iterations, total_tool_calls)
}

/// Agent rescue pass: investigates findings that were killed by blind validation.
/// Only rescues ones it can confirm with cross-file tool evidence.
async fn rescue_agentic(
    state: &AppState,
    ctx: &AgentContext,
    killed_findings: &[Finding],
) -> (Vec<Finding>, usize, usize) {
    let tools = build_tool_definitions();
    let instructions = prompts::SYSTEM_AGENT_RESCUE;
    let initial_prompt = prompts::build_agent_rescue_prompt(ctx, killed_findings);

    let mut resp = match call_responses_api(
        state,
        instructions,
        Some(&initial_prompt),
        &tools,
        None,
        None,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("Agent rescue initial call failed: {e}");
            return (Vec::new(), 0, 0);
        }
    };

    let max_iterations = 20;
    let mut total_tool_calls = 0usize;
    let mut iterations = 0usize;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            warn!("Agent rescue hit max iterations ({max_iterations})");
            break;
        }

        match parse_agent_step(&resp.output) {
            AgentStep::Finished(findings) => {
                info!(
                    "Agent rescue finished: {} rescued after {} iterations, {} tool calls",
                    findings.len(), iterations, total_tool_calls
                );
                return (findings, iterations, total_tool_calls);
            }
            AgentStep::ToolCalls(calls) => {
                total_tool_calls += calls.len();
                info!(
                    "Agent rescue iter {}: {} tool calls (total: {})",
                    iterations, calls.len(), total_tool_calls
                );

                let mut tool_outputs = Vec::new();
                for (name, arguments, call_id) in &calls {
                    let result = match name.as_str() {
                        "get_entity" => execute_get_entity(ctx, arguments),
                        "read_file" => execute_read_file(state, ctx, arguments).await,
                        "search_code" => execute_search_code(state, ctx, arguments).await,
                        "get_dependents" => execute_get_dependents(ctx, arguments),
                        "get_dependencies" => execute_get_dependencies(ctx, arguments),
                        _ => format!(r#"{{"error": "unknown tool: {name}"}}"#),
                    };

                    tool_outputs.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }));
                }

                let prev_id = resp.id.clone();
                resp = match call_responses_api(
                    state,
                    instructions,
                    None,
                    &tools,
                    Some(&prev_id),
                    Some(tool_outputs),
                )
                .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        warn!("Agent rescue continuation failed: {e}");
                        break;
                    }
                };
            }
            AgentStep::Evidence(_) => {
                info!("Agent rescue got evidence instead of findings");
                break;
            }
            AgentStep::KeptIndices(_) => {
                info!("Agent rescue got kept_indices unexpectedly");
                break;
            }
            AgentStep::Empty => {
                info!("Agent rescue returned empty");
                break;
            }
        }
    }

    (Vec::new(), iterations, total_tool_calls)
}

/// v14 rescue: agent investigates killed findings and gathers evidence,
/// then blind validator re-evaluates each finding WITH the evidence attached.
async fn rescue_with_evidence(
    state: &AppState,
    ctx: &AgentContext,
    killed_findings: &[Finding],
    pr_title: &str,
    diff: &str,
) -> (Vec<Finding>, usize, usize) {
    // Step 1: Agent investigates and gathers evidence (no judgment)
    let tools = build_evidence_tool_definitions();
    let instructions = prompts::SYSTEM_AGENT_EVIDENCE;
    let initial_prompt = prompts::build_agent_evidence_prompt(ctx, killed_findings);

    let mut resp = match call_responses_api(
        state,
        instructions,
        Some(&initial_prompt),
        &tools,
        None,
        None,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("Agent evidence initial call failed: {e}");
            return (Vec::new(), 0, 0);
        }
    };

    let max_iterations = 20;
    let mut total_tool_calls = 0usize;
    let mut iterations = 0usize;
    let mut evidence_reports: Vec<EvidenceReport> = Vec::new();

    loop {
        iterations += 1;
        if iterations > max_iterations {
            warn!("Agent evidence hit max iterations ({max_iterations})");
            break;
        }

        match parse_agent_step(&resp.output) {
            AgentStep::Evidence(reports) => {
                info!(
                    "Agent evidence finished: {} reports after {} iterations, {} tool calls",
                    reports.len(), iterations, total_tool_calls
                );
                evidence_reports = reports;
                break;
            }
            AgentStep::Finished(_) => {
                // Agent called submit_findings instead of submit_evidence, treat as empty
                warn!("Agent called submit_findings instead of submit_evidence");
                break;
            }
            AgentStep::ToolCalls(calls) => {
                total_tool_calls += calls.len();
                info!(
                    "Agent evidence iter {}: {} tool calls (total: {})",
                    iterations, calls.len(), total_tool_calls
                );

                let mut tool_outputs = Vec::new();
                for (name, arguments, call_id) in &calls {
                    let result = match name.as_str() {
                        "get_entity" => execute_get_entity(ctx, arguments),
                        "read_file" => execute_read_file(state, ctx, arguments).await,
                        "search_code" => execute_search_code(state, ctx, arguments).await,
                        "get_dependents" => execute_get_dependents(ctx, arguments),
                        "get_dependencies" => execute_get_dependencies(ctx, arguments),
                        _ => format!(r#"{{"error": "unknown tool: {name}"}}"#),
                    };

                    tool_outputs.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }));
                }

                let prev_id = resp.id.clone();
                resp = match call_responses_api(
                    state,
                    instructions,
                    None,
                    &tools,
                    Some(&prev_id),
                    Some(tool_outputs),
                )
                .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        warn!("Agent evidence continuation failed: {e}");
                        break;
                    }
                };
            }
            AgentStep::KeptIndices(_) => {
                info!("Agent evidence got kept_indices unexpectedly");
                break;
            }
            AgentStep::Empty => {
                info!("Agent evidence returned empty");
                break;
            }
        }
    }

    // Filter to only reports where agent recommends rescue
    let rescue_candidates: Vec<&EvidenceReport> = evidence_reports
        .iter()
        .filter(|r| r.verdict == "rescue")
        .collect();

    info!(
        "FUNNEL agent evidence: {} reports, {} recommend rescue",
        evidence_reports.len(), rescue_candidates.len()
    );

    if rescue_candidates.is_empty() {
        return (Vec::new(), iterations, total_tool_calls);
    }

    // Step 2: Re-validate rescued findings with evidence through blind validator
    let truncated_diff = prompts::truncate_diff(diff, 40_000);

    // Build findings with evidence appended
    let enriched_findings: Vec<Finding> = rescue_candidates
        .iter()
        .filter_map(|report| {
            killed_findings.get(report.finding_index).map(|original| {
                let mut enriched_evidence = String::new();
                if let Some(ref ev) = original.evidence {
                    enriched_evidence.push_str(ev);
                    enriched_evidence.push_str("\n\n");
                }
                enriched_evidence.push_str("Cross-file evidence from investigation:\n");
                enriched_evidence.push_str(&report.evidence_summary);
                for snippet in &report.code_snippets {
                    enriched_evidence.push_str("\n```\n");
                    enriched_evidence.push_str(snippet);
                    enriched_evidence.push_str("\n```");
                }

                Finding {
                    issue: original.issue.clone(),
                    evidence: Some(enriched_evidence),
                    severity: original.severity.clone(),
                    file: original.file.clone(),
                }
            })
        })
        .collect();

    info!(
        "FUNNEL revalidating {} enriched findings through blind validator",
        enriched_findings.len()
    );

    // Run through the same strict blind validator
    let revalidated = match validate_findings_seeded(
        state,
        pr_title,
        &truncated_diff,
        &enriched_findings,
        Some(42),
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            warn!("Revalidation of rescued findings failed: {e}");
            return (Vec::new(), iterations, total_tool_calls);
        }
    };

    info!(
        "FUNNEL revalidation: {} -> {} survived",
        enriched_findings.len(), revalidated.len()
    );

    (revalidated, iterations, total_tool_calls)
}

/// Build tool definitions for evidence-gathering agent (no submit_findings, has submit_evidence).
fn build_evidence_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "name": "get_entity",
            "description": "Get full details of a changed entity from triage. Returns before/after code, risk score, blast radius, dependents, dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "read_file",
            "description": "Read a file from the repo at head or base commit. Use for full context, imports, type definitions, surrounding code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path relative to repo root" },
                    "ref": { "type": "string", "enum": ["head", "base"], "description": "Which commit to read from" },
                    "start_line": { "type": "integer", "description": "Optional 1-indexed start line" },
                    "end_line": { "type": "integer", "description": "Optional end line, max 200 lines from start" }
                },
                "required": ["path", "ref"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "search_code",
            "description": "Search the repo for code matching a query. Find callers, implementations, usages of a function or type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query (function name, type name, etc.)" },
                    "path_prefix": { "type": "string", "description": "Optional path filter like 'src/' or 'lib/'" }
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "get_dependents",
            "description": "Get entities that depend on (call/reference) a given entity. Shows blast radius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "get_dependencies",
            "description": "Get entities that a given entity depends on (calls/imports). Shows what this entity relies on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "submit_evidence",
            "description": "Submit your evidence reports for each investigated finding. Do NOT decide if findings are valid. Just report what you found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reports": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "finding_index": { "type": "integer", "description": "0-indexed position of the finding in the killed list" },
                                "finding_issue": { "type": "string", "description": "The original finding text (for reference)" },
                                "evidence_summary": { "type": "string", "description": "What you found in the codebase. Be factual: quote actual code, function signatures, caller patterns. No opinions." },
                                "code_snippets": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "Relevant code snippets you found with tools that relate to this finding"
                                },
                                "verdict": { "type": "string", "enum": ["rescue", "reject"], "description": "rescue = you found concrete cross-file evidence supporting the bug. reject = no evidence found or issue is speculative." }
                            },
                            "required": ["finding_index", "finding_issue", "evidence_summary", "code_snippets", "verdict"]
                        }
                    }
                },
                "required": ["reports"]
            }
        }),
    ]
}

// --- Responses API for agentic review ---

/// Call OpenAI Responses API. Initial call uses instructions + input.
/// Continuation calls use previous_response_id + tool results.
async fn call_responses_api(
    state: &AppState,
    instructions: &str,
    input: Option<&str>,
    tools: &[serde_json::Value],
    previous_response_id: Option<&str>,
    tool_outputs: Option<Vec<serde_json::Value>>,
) -> Result<ResponsesResponse, String> {
    let mut body = serde_json::json!({
        "model": state.openai_model,
        "instructions": instructions,
        "tools": tools,
        "temperature": 0.0,
    });

    if let Some(prev_id) = previous_response_id {
        body["previous_response_id"] = serde_json::json!(prev_id);
        if let Some(outputs) = tool_outputs {
            body["input"] = serde_json::Value::Array(outputs);
        }
    } else if let Some(inp) = input {
        body["input"] = serde_json::json!(inp);
    }

    let resp = state
        .http
        .post("https://api.openai.com/v1/responses")
        .header("Authorization", format!("Bearer {}", state.openai_api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("responses API request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Responses API error {status}: {text}"));
    }

    resp.json::<ResponsesResponse>()
        .await
        .map_err(|e| format!("parse responses failed: {e}"))
}

/// Parse agent response into tool calls, finished findings, or empty.
fn parse_agent_step(output: &[ResponseOutput]) -> AgentStep {
    let mut tool_calls = Vec::new();

    for item in output {
        match item {
            ResponseOutput::FunctionCall {
                name,
                arguments,
                call_id,
            } => {
                if name == "submit_kept_indices" {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(arguments) {
                        let indices = parsed
                            .get("kept_indices")
                            .and_then(|f| f.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                                    .collect()
                            })
                            .unwrap_or_default();
                        return AgentStep::KeptIndices(indices);
                    }
                    return AgentStep::KeptIndices(Vec::new());
                }
                if name == "submit_findings" {
                    // Parse findings from arguments
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(arguments) {
                        let findings = parsed
                            .get("findings")
                            .and_then(|f| f.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| {
                                        let issue = v.get("issue")?.as_str()?.to_string();
                                        Some(Finding {
                                            issue,
                                            evidence: v
                                                .get("evidence")
                                                .and_then(|e| e.as_str())
                                                .map(String::from),
                                            severity: v
                                                .get("severity")
                                                .and_then(|s| s.as_str())
                                                .map(String::from),
                                            file: v
                                                .get("file")
                                                .and_then(|f| f.as_str())
                                                .map(String::from),
                                        })
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        return AgentStep::Finished(findings);
                    }
                    return AgentStep::Finished(Vec::new());
                }
                if name == "submit_evidence" {
                    // Parse evidence reports from arguments
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(arguments) {
                        let reports = parsed
                            .get("reports")
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| {
                                        Some(EvidenceReport {
                                            finding_index: v.get("finding_index")?.as_u64()? as usize,
                                            finding_issue: v.get("finding_issue")?.as_str()?.to_string(),
                                            evidence_summary: v.get("evidence_summary")?.as_str()?.to_string(),
                                            code_snippets: v.get("code_snippets")
                                                .and_then(|s| s.as_array())
                                                .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                                                .unwrap_or_default(),
                                            verdict: v.get("verdict")?.as_str()?.to_string(),
                                        })
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        return AgentStep::Evidence(reports);
                    }
                    return AgentStep::Evidence(Vec::new());
                }
                tool_calls.push((name.clone(), arguments.clone(), call_id.clone()));
            }
            ResponseOutput::Message { .. } => {}
        }
    }

    if tool_calls.is_empty() {
        AgentStep::Empty
    } else {
        AgentStep::ToolCalls(tool_calls)
    }
}

/// Build JSON tool definitions for the 6 agent tools.
fn build_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "name": "get_entity",
            "description": "Get full details of a changed entity from triage. Returns before/after code, risk score, blast radius, dependents, dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "read_file",
            "description": "Read a file from the repo at head or base commit. Use for full context, imports, type definitions, surrounding code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path relative to repo root" },
                    "ref": { "type": "string", "enum": ["head", "base"], "description": "Which commit to read from" },
                    "start_line": { "type": "integer", "description": "Optional 1-indexed start line" },
                    "end_line": { "type": "integer", "description": "Optional end line, max 200 lines from start" }
                },
                "required": ["path", "ref"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "search_code",
            "description": "Search the repo for code matching a query. Find callers, implementations, usages of a function or type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query (function name, type name, etc.)" },
                    "path_prefix": { "type": "string", "description": "Optional path filter like 'src/' or 'lib/'" }
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "get_dependents",
            "description": "Get entities that depend on (call/reference) a given entity. Shows blast radius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "get_dependencies",
            "description": "Get entities that a given entity depends on (calls/imports). Shows what this entity relies on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "submit_findings",
            "description": "Submit your final list of confirmed bug findings and end the review. Only call this once when done investigating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "issue": { "type": "string", "description": "Clear description of the bug" },
                                "evidence": { "type": "string", "description": "Specific code/reasoning proving this is a bug" },
                                "severity": { "type": "string", "enum": ["critical", "high", "medium", "low"] },
                                "file": { "type": "string", "description": "File path where the bug is" }
                            },
                            "required": ["issue", "evidence", "severity", "file"]
                        }
                    }
                },
                "required": ["findings"]
            }
        }),
    ]
}

/// Build tool definitions for challenge agent: same tools but submit_kept_indices instead of submit_findings.
/// The agent returns which findings to KEEP by their 1-indexed number, preserving original text.
fn build_challenge_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "name": "get_entity",
            "description": "Get full details of a changed entity from triage. Returns before/after code, risk score, blast radius, dependents, dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "read_file",
            "description": "Read a file from the repo at head or base commit. Use for full context, imports, type definitions, surrounding code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path relative to repo root" },
                    "ref": { "type": "string", "enum": ["head", "base"], "description": "Which commit to read from" },
                    "start_line": { "type": "integer", "description": "Optional 1-indexed start line" },
                    "end_line": { "type": "integer", "description": "Optional end line, max 200 lines from start" }
                },
                "required": ["path", "ref"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "search_code",
            "description": "Search the repo for code matching a query. Find callers, implementations, usages of a function or type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query (function name, type name, etc.)" },
                    "path_prefix": { "type": "string", "description": "Optional path filter like 'src/' or 'lib/'" }
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "get_dependents",
            "description": "Get entities that depend on (call/reference) a given entity. Shows blast radius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "get_dependencies",
            "description": "Get entities that a given entity depends on (calls/imports). Shows what this entity relies on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "submit_kept_indices",
            "description": "Submit the indices (1-indexed numbers) of findings you want to KEEP. Only call this once when done investigating. Do NOT rewrite findings - just return their numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kept_indices": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "List of 1-indexed finding numbers to keep (e.g. [1, 3, 5] keeps findings #1, #3, #5)"
                    }
                },
                "required": ["kept_indices"]
            }
        }),
    ]
}

// --- Tool executors ---

fn execute_get_entity(ctx: &AgentContext, args: &str) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return r#"{"error": "invalid arguments"}"#.to_string(),
    };

    let name = parsed
        .get("entity_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let file_filter = parsed.get("file_path").and_then(|v| v.as_str());

    let entity = ctx.entity_reviews.iter().find(|e| {
        let name_match = e.entity_name == name
            || e.entity_name.ends_with(&format!("::{name}"))
            || e.entity_name.ends_with(&format!(".{name}"));
        let file_match = file_filter.map_or(true, |f| e.file_path.contains(f));
        name_match && file_match
    });

    match entity {
        Some(e) => {
            let before: String = e
                .before_content
                .as_ref()
                .map(|s| s.chars().take(3000).collect())
                .unwrap_or_default();
            let after: String = e
                .after_content
                .as_ref()
                .map(|s| s.chars().take(3000).collect())
                .unwrap_or_default();

            let dependents: Vec<serde_json::Value> = e
                .dependent_names
                .iter()
                .take(15)
                .map(|(n, f)| serde_json::json!({"name": n, "file": f}))
                .collect();
            let dependencies: Vec<serde_json::Value> = e
                .dependency_names
                .iter()
                .take(15)
                .map(|(n, f)| serde_json::json!({"name": n, "file": f}))
                .collect();

            serde_json::json!({
                "entity_name": e.entity_name,
                "entity_type": e.entity_type,
                "file_path": e.file_path,
                "change_type": format!("{:?}", e.change_type),
                "risk_score": e.risk_score,
                "blast_radius": e.blast_radius,
                "is_public_api": e.is_public_api,
                "before_content": before,
                "after_content": after,
                "dependents": dependents,
                "dependencies": dependencies,
            })
            .to_string()
        }
        None => {
            // List available entities to help the agent
            let available: Vec<String> = ctx
                .entity_reviews
                .iter()
                .take(30)
                .map(|e| format!("{} ({})", e.entity_name, e.file_path))
                .collect();
            serde_json::json!({
                "error": "entity not found",
                "available": available,
            })
            .to_string()
        }
    }
}

async fn execute_read_file(
    state: &AppState,
    ctx: &AgentContext,
    args: &str,
) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return r#"{"error": "invalid arguments"}"#.to_string(),
    };

    let path = match parsed.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return r#"{"error": "path is required"}"#.to_string(),
    };

    let sha = match parsed.get("ref").and_then(|v| v.as_str()) {
        Some("base") => &ctx.base_sha,
        _ => &ctx.head_sha,
    };

    let start_line = parsed
        .get("start_line")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let end_line = parsed
        .get("end_line")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    let encoded_path = urlencoding::encode(path);
    let url = format!(
        "https://api.github.com/repos/{}/contents/{}?ref={}",
        ctx.repo, encoded_path, sha
    );

    let resp = match state
        .http
        .get(&url)
        .header("Authorization", format!("token {}", state.github_token))
        .header("Accept", "application/vnd.github.v3+json")
        .header("User-Agent", "inspect-api")
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!(r#"{{"error": "request failed: {e}"}}"#),
    };

    if !resp.status().is_success() {
        let status = resp.status();
        return format!(r#"{{"error": "GitHub API {status} for {path}"}}"#);
    }

    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return format!(r#"{{"error": "parse failed: {e}"}}"#),
    };

    let content_b64 = body
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // GitHub returns base64 with newlines
    let cleaned: String = content_b64.chars().filter(|c| !c.is_whitespace()).collect();
    let decoded = match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &cleaned)
    {
        Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
        Err(_) => return r#"{"error": "base64 decode failed"}"#.to_string(),
    };

    // Apply line range
    let lines: Vec<&str> = decoded.lines().collect();
    let start = start_line.unwrap_or(1).saturating_sub(1).min(lines.len());
    let end = end_line
        .unwrap_or(lines.len())
        .min(start + 200)
        .min(lines.len())
        .max(start);

    let selected: String = lines[start..end].join("\n");

    serde_json::json!({
        "path": path,
        "lines": format!("{}-{}", start + 1, end),
        "content": selected,
    })
    .to_string()
}

async fn execute_search_code(
    state: &AppState,
    ctx: &AgentContext,
    args: &str,
) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return r#"{"error": "invalid arguments"}"#.to_string(),
    };

    let query = match parsed.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return r#"{"error": "query is required"}"#.to_string(),
    };

    let path_prefix = parsed.get("path_prefix").and_then(|v| v.as_str());

    let mut search_q = format!("{query} repo:{}", ctx.repo);
    if let Some(prefix) = path_prefix {
        search_q.push_str(&format!(" path:{prefix}"));
    }

    let url = format!(
        "https://api.github.com/search/code?q={}&per_page=20",
        urlencoding::encode(&search_q)
    );

    let resp = match state
        .http
        .get(&url)
        .header("Authorization", format!("token {}", state.github_token))
        .header("Accept", "application/vnd.github.v3+json")
        .header("User-Agent", "inspect-api")
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!(r#"{{"error": "search failed: {e}"}}"#),
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return format!(r#"{{"error": "GitHub search API {status}: {text}"}}"#);
    }

    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return format!(r#"{{"error": "parse failed: {e}"}}"#),
    };

    let items = body
        .get("items")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let results: Vec<serde_json::Value> = items
        .iter()
        .take(20)
        .filter_map(|item| {
            let path = item.get("path")?.as_str()?;
            let name = item.get("name")?.as_str()?;
            Some(serde_json::json!({
                "path": path,
                "name": name,
            }))
        })
        .collect();

    serde_json::json!({
        "total_count": body.get("total_count").and_then(|v| v.as_u64()).unwrap_or(0),
        "results": results,
    })
    .to_string()
}

fn execute_get_dependents(ctx: &AgentContext, args: &str) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return r#"{"error": "invalid arguments"}"#.to_string(),
    };

    let name = parsed
        .get("entity_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let file_filter = parsed.get("file_path").and_then(|v| v.as_str());

    let entity = ctx.entity_reviews.iter().find(|e| {
        let name_match = e.entity_name == name
            || e.entity_name.ends_with(&format!("::{name}"))
            || e.entity_name.ends_with(&format!(".{name}"));
        let file_match = file_filter.map_or(true, |f| e.file_path.contains(f));
        name_match && file_match
    });

    match entity {
        Some(e) => {
            let dependents: Vec<serde_json::Value> = e
                .dependent_names
                .iter()
                .take(20)
                .map(|(n, f)| serde_json::json!({"name": n, "file": f}))
                .collect();
            serde_json::json!({
                "entity": e.entity_name,
                "dependent_count": e.dependent_count,
                "blast_radius": e.blast_radius,
                "dependents": dependents,
            })
            .to_string()
        }
        None => r#"{"error": "entity not found in triage"}"#.to_string(),
    }
}

fn execute_get_dependencies(ctx: &AgentContext, args: &str) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return r#"{"error": "invalid arguments"}"#.to_string(),
    };

    let name = parsed
        .get("entity_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let file_filter = parsed.get("file_path").and_then(|v| v.as_str());

    let entity = ctx.entity_reviews.iter().find(|e| {
        let name_match = e.entity_name == name
            || e.entity_name.ends_with(&format!("::{name}"))
            || e.entity_name.ends_with(&format!(".{name}"));
        let file_match = file_filter.map_or(true, |f| e.file_path.contains(f));
        name_match && file_match
    });

    match entity {
        Some(e) => {
            let dependencies: Vec<serde_json::Value> = e
                .dependency_names
                .iter()
                .take(20)
                .map(|(n, f)| serde_json::json!({"name": n, "file": f}))
                .collect();
            serde_json::json!({
                "entity": e.entity_name,
                "dependency_count": e.dependency_count,
                "dependencies": dependencies,
            })
            .to_string()
        }
        None => r#"{"error": "entity not found in triage"}"#.to_string(),
    }
}

// --- Agentic review v2 ---

/// Agentic review: LLM iteratively investigates with tools, then submits findings.
pub async fn review_agentic_v2(
    state: &AppState,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    let tools = build_tool_definitions();
    let instructions = prompts::SYSTEM_AGENT_REVIEW;
    let initial_prompt = prompts::build_agent_initial_prompt(ctx);

    // Initial call
    let mut resp = match call_responses_api(
        state,
        instructions,
        Some(&initial_prompt),
        &tools,
        None,
        None,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("Agentic initial call failed: {e}");
            return (Vec::new(), 0, 0);
        }
    };

    let max_iterations = 15;
    let mut total_tool_calls = 0usize;
    let mut iterations = 0usize;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            warn!("Agentic review hit max iterations ({max_iterations})");
            break;
        }

        match parse_agent_step(&resp.output) {
            AgentStep::Finished(findings) => {
                info!(
                    "Agentic review finished: {} findings after {} iterations, {} tool calls",
                    findings.len(),
                    iterations,
                    total_tool_calls
                );
                return (findings, iterations, total_tool_calls);
            }
            AgentStep::ToolCalls(calls) => {
                total_tool_calls += calls.len();
                info!(
                    "Agentic iteration {}: {} tool calls (total: {})",
                    iterations,
                    calls.len(),
                    total_tool_calls
                );

                let mut tool_outputs = Vec::new();
                for (name, arguments, call_id) in &calls {
                    let result = match name.as_str() {
                        "get_entity" => execute_get_entity(ctx, arguments),
                        "read_file" => execute_read_file(state, ctx, arguments).await,
                        "search_code" => execute_search_code(state, ctx, arguments).await,
                        "get_dependents" => execute_get_dependents(ctx, arguments),
                        "get_dependencies" => execute_get_dependencies(ctx, arguments),
                        _ => format!(r#"{{"error": "unknown tool: {name}"}}"#),
                    };

                    tool_outputs.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }));
                }

                // Continue with tool results
                let prev_id = resp.id.clone();
                resp = match call_responses_api(
                    state,
                    instructions,
                    None,
                    &tools,
                    Some(&prev_id),
                    Some(tool_outputs),
                )
                .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        warn!("Agentic continuation failed: {e}");
                        break;
                    }
                };
            }
            AgentStep::Evidence(_) => {
                info!("Agentic review got evidence instead of findings");
                break;
            }
            AgentStep::KeptIndices(_) => {
                info!("Agentic review got kept_indices unexpectedly");
                break;
            }
            AgentStep::Empty => {
                info!("Agentic review returned empty (no tool calls, no findings)");
                break;
            }
        }
    }

    (Vec::new(), iterations, total_tool_calls)
}

/// Agentic challenge: agent tries to DISPROVE each candidate finding using tools.
/// Opposite of validate_agentic: starts skeptical, only keeps what it can't disprove.
async fn challenge_agentic(
    state: &AppState,
    ctx: &AgentContext,
    candidates: &[Finding],
    max_findings: usize,
) -> (Vec<Finding>, usize, usize) {
    let tools = build_challenge_tool_definitions();
    let instructions = prompts::SYSTEM_AGENT_CHALLENGE;
    let initial_prompt = prompts::build_agent_challenge_prompt(ctx, candidates);

    let mut resp = match call_responses_api(
        state,
        instructions,
        Some(&initial_prompt),
        &tools,
        None,
        None,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("Agentic challenge initial call failed: {e}");
            return (candidates.iter().take(max_findings).cloned().collect(), 0, 0);
        }
    };

    // Scale iterations with candidate count (more findings = more tool calls needed)
    let max_iterations = if candidates.len() > 20 { 50 } else { 25 };
    let mut total_tool_calls = 0usize;
    let mut iterations = 0usize;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            warn!("Agentic challenge hit max iterations ({max_iterations})");
            break;
        }

        match parse_agent_step(&resp.output) {
            AgentStep::KeptIndices(indices) => {
                // Map 1-indexed indices back to original candidates
                let kept: Vec<Finding> = indices
                    .iter()
                    .filter_map(|&i| candidates.get(i.wrapping_sub(1)).cloned())
                    .collect();
                info!(
                    "Agentic challenge finished (index-based): kept {}/{} (indices: {:?}) after {} iterations, {} tool calls",
                    kept.len(), candidates.len(), indices, iterations, total_tool_calls
                );
                return (kept.into_iter().take(max_findings).collect(), iterations, total_tool_calls);
            }
            AgentStep::Finished(findings) => {
                // Fallback: agent called submit_findings instead of submit_kept_indices
                warn!("Agentic challenge used submit_findings instead of submit_kept_indices, using text match fallback");
                return (findings.into_iter().take(max_findings).collect(), iterations, total_tool_calls);
            }
            AgentStep::ToolCalls(calls) => {
                total_tool_calls += calls.len();
                info!(
                    "Agentic challenge iter {}: {} tool calls (total: {})",
                    iterations, calls.len(), total_tool_calls
                );

                let mut tool_outputs = Vec::new();
                for (name, arguments, call_id) in &calls {
                    let result = match name.as_str() {
                        "get_entity" => execute_get_entity(ctx, arguments),
                        "read_file" => execute_read_file(state, ctx, arguments).await,
                        "search_code" => execute_search_code(state, ctx, arguments).await,
                        "get_dependents" => execute_get_dependents(ctx, arguments),
                        "get_dependencies" => execute_get_dependencies(ctx, arguments),
                        _ => format!(r#"{{"error": "unknown tool: {name}"}}"#),
                    };

                    tool_outputs.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }));
                }

                let prev_id = resp.id.clone();
                resp = match call_responses_api(
                    state,
                    instructions,
                    None,
                    &tools,
                    Some(&prev_id),
                    Some(tool_outputs),
                )
                .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        warn!("Agentic challenge continuation failed: {e}");
                        break;
                    }
                };
            }
            AgentStep::Evidence(_) => {
                info!("Agentic challenge got evidence instead of findings");
                break;
            }
            AgentStep::Empty => {
                info!("Agentic challenge returned empty");
                break;
            }
        }
    }

    warn!("Agentic challenge did not submit findings, returning candidates as fallback");
    (candidates.iter().take(max_findings).cloned().collect(), iterations, total_tool_calls)
}

/// Hybrid v20: v10 pipeline + agentic challenge pass.
/// Runs the standard 9-lens + blind validation (v10), then passes survivors
/// through an agent with tools that tries to disprove each finding.
pub async fn review_hybrid_v20(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    // Step 1: Run v10 pipeline
    let v10_results = review_hybrid_inner(state, pr_title, diff, triage_section, max_findings, None).await;

    if v10_results.is_empty() {
        return (v10_results, 0, 0);
    }

    info!("v20: v10 produced {} findings, sending to agentic challenge", v10_results.len());

    // Step 2: Agentic challenge - try to disprove each finding
    let (challenged, iterations, tool_calls) = challenge_agentic(state, ctx, &v10_results, max_findings).await;

    info!(
        "v20: agentic challenge {} -> {} ({} dropped, {} iterations, {} tool calls)",
        v10_results.len(), challenged.len(), v10_results.len() - challenged.len(), iterations, tool_calls
    );

    (challenged, iterations, tool_calls)
}

/// Hybrid v21: lenient enriched validation (v16-style) + agentic challenge (v20-style).
/// The enriched validator is more lenient (passes more TPs through), then the
/// agentic challenge cleans up FPs with better selectivity.
pub async fn review_hybrid_v21(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    // Step 1: Run v16 pipeline (enriched lenient validation)
    let v16_results = review_hybrid_v16(state, pr_title, diff, triage_section, max_findings * 2, &ctx.entity_reviews).await;

    if v16_results.is_empty() {
        return (v16_results, 0, 0);
    }

    info!("v21: lenient validator produced {} findings, sending to agentic challenge", v16_results.len());

    // Step 2: Agentic challenge - try to disprove each finding
    let (challenged, iterations, tool_calls) = challenge_agentic(state, ctx, &v16_results, max_findings).await;

    info!(
        "v21: agentic challenge {} -> {} ({} dropped, {} iterations, {} tool calls)",
        v16_results.len(), challenged.len(), v16_results.len() - challenged.len(), iterations, tool_calls
    );

    (challenged, iterations, tool_calls)
}

/// Hybrid v22: raw lenses + agentic challenge (no blind validator).
/// Skips the blind validator entirely. Sends all deduped+filtered lens output
/// straight to the agentic challenge, which can read source code to decide.
pub async fn review_hybrid_v22(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    // Step 1: Raw lenses (dedup + structural filter, no validation)
    let raw = review_raw_lenses(state, pr_title, diff, triage_section, 50).await;

    if raw.is_empty() {
        return (raw, 0, 0);
    }

    info!("v22: raw lenses produced {} findings, sending directly to agentic challenge", raw.len());

    // Step 2: Agentic challenge on all raw findings
    let (challenged, iterations, tool_calls) = challenge_agentic(state, ctx, &raw, max_findings).await;

    info!(
        "v22: agentic challenge {} -> {} ({} dropped, {} iterations, {} tool calls)",
        raw.len(), challenged.len(), raw.len() - challenged.len(), iterations, tool_calls
    );

    (challenged, iterations, tool_calls)
}

/// Hybrid v23: two validator passes (union) + agentic challenge.
/// Runs blind validation twice with different seeds, takes the union of survivors.
/// A TP only dies if BOTH validators reject it. Then agentic challenge cleans FPs.
pub async fn review_hybrid_v23(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    // Step 1: Run 9 lenses + dedup + structural filter
    let raw = review_raw_lenses(state, pr_title, diff, triage_section, 50).await;

    if raw.is_empty() {
        return (raw, 0, 0);
    }

    if raw.len() <= 2 {
        info!("v23: only {} findings, skipping validation", raw.len());
        return (raw, 0, 0);
    }

    let truncated = prompts::truncate_diff(diff, 65_000);

    // Step 2: Two validator passes in parallel with different seeds
    let (v1, v2) = tokio::join!(
        validate_findings_seeded(state, pr_title, &truncated, &raw, Some(42)),
        validate_findings_seeded(state, pr_title, &truncated, &raw, Some(123)),
    );

    // Union: keep a finding if EITHER validator kept it
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut union_findings: Vec<Finding> = Vec::new();

    let v1_findings = v1.unwrap_or_default();
    let v2_findings = v2.unwrap_or_default();

    for f in v1_findings.iter().chain(v2_findings.iter()) {
        let key: String = f.issue.to_lowercase().chars().take(80).collect();
        if seen.insert(key) {
            union_findings.push(f.clone());
        }
    }

    info!(
        "v23: raw={}, validator1={}, validator2={}, union={}",
        raw.len(), v1_findings.len(), v2_findings.len(), union_findings.len()
    );

    if union_findings.is_empty() {
        return (union_findings, 0, 0);
    }

    // Step 3: Agentic challenge on the union
    let (challenged, iterations, tool_calls) = challenge_agentic(state, ctx, &union_findings, max_findings).await;

    info!(
        "v23: agentic challenge {} -> {} ({} dropped, {} iterations, {} tool calls)",
        union_findings.len(), challenged.len(), union_findings.len() - challenged.len(), iterations, tool_calls
    );

    (challenged, iterations, tool_calls)
}

/// Hybrid v24: cross-model validation (GPT-5.4 + Claude Sonnet 4.6 union) + agentic challenge.
/// Different models have different systematic biases, so their union recovers TPs
/// that a single model would kill. The agentic challenge cleans up FPs.
pub async fn review_hybrid_v24(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    // Step 1: Run 9 lenses + dedup + structural filter
    let raw = review_raw_lenses(state, pr_title, diff, triage_section, 50).await;

    if raw.is_empty() {
        return (raw, 0, 0);
    }

    if raw.len() <= 2 {
        info!("v24: only {} findings, skipping validation", raw.len());
        return (raw, 0, 0);
    }

    let truncated = prompts::truncate_diff(diff, 65_000);

    // Step 2: Cross-model validation in parallel (GPT-5.4 + Claude Sonnet 4.6)
    let (v_gpt, v_claude) = tokio::join!(
        validate_findings_gpt54(state, pr_title, &truncated, &raw),
        validate_findings_sonnet46(state, pr_title, &truncated, &raw),
    );

    // Union: keep a finding if EITHER model kept it
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut union_findings: Vec<Finding> = Vec::new();

    let gpt_findings = match v_gpt {
        Ok(f) => f,
        Err(e) => { warn!("v24: GPT-5.4 validator failed: {e}"); Vec::new() }
    };
    let claude_findings = match v_claude {
        Ok(f) => f,
        Err(e) => { warn!("v24: Sonnet 4.6 validator failed: {e}"); Vec::new() }
    };

    for f in gpt_findings.iter().chain(claude_findings.iter()) {
        let key: String = f.issue.to_lowercase().chars().take(80).collect();
        if seen.insert(key) {
            union_findings.push(f.clone());
        }
    }

    info!(
        "v24: raw={}, gpt54={}, sonnet46={}, union={}",
        raw.len(), gpt_findings.len(), claude_findings.len(), union_findings.len()
    );

    if union_findings.is_empty() {
        return (union_findings, 0, 0);
    }

    // Step 3: Agentic challenge on the union
    let (challenged, iterations, tool_calls) = challenge_agentic(state, ctx, &union_findings, max_findings).await;

    info!(
        "v24: agentic challenge {} -> {} ({} dropped, {} iterations, {} tool calls)",
        union_findings.len(), challenged.len(), union_findings.len() - challenged.len(), iterations, tool_calls
    );

    (challenged, iterations, tool_calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_issues_string_array() {
        let input = r#"{"issues": ["bug 1", "bug 2"]}"#;
        let findings = parse_issues(input);
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].issue, "bug 1");
    }

    #[test]
    fn test_parse_issues_object_array() {
        let input = r#"{"issues": [{"issue": "null check missing", "evidence": "if (x)"}]}"#;
        let findings = parse_issues(input);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].issue, "null check missing");
        assert_eq!(findings[0].evidence.as_deref(), Some("if (x)"));
    }

    #[test]
    fn test_parse_issues_with_code_fence() {
        let input = "```json\n{\"issues\": [\"bug\"]}\n```";
        let findings = parse_issues(input);
        assert_eq!(findings.len(), 1);
    }

    #[test]
    fn test_strip_code_fences() {
        assert_eq!(strip_code_fences("```json\n{}\n```"), "{}");
        assert_eq!(strip_code_fences("```\n{}\n```"), "{}");
        assert_eq!(strip_code_fences("{}"), "{}");
    }
}
