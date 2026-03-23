use serde::{Deserialize, Serialize};
use tracing::warn;

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

/// Strip markdown code fences and parse JSON issues.
fn parse_issues(text: &str) -> Vec<Finding> {
    let cleaned = strip_code_fences(text);

    let parsed: Result<IssuesResponse, _> = serde_json::from_str(&cleaned);
    match parsed {
        Ok(resp) => resp
            .issues
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
            .collect(),
        Err(e) => {
            warn!("Failed to parse LLM response as JSON: {e}");
            Vec::new()
        }
    }
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

/// Hybrid v10 strategy: 9 parallel lenses + structural filter + validation.
pub async fn review_hybrid_v10(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
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

    // 9 lenses in parallel
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

    // Merge + dedup
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
        } else {
            warn!("Lens failed: {:?}", result.err());
        }
    }

    if all_findings.is_empty() {
        return Vec::new();
    }

    // Structural file filter
    all_findings = structural_file_filter(all_findings, &diff_files);

    if all_findings.is_empty() {
        return Vec::new();
    }

    // Skip validation if few findings
    if all_findings.len() <= 2 {
        return all_findings;
    }

    // Validation pass with seed=42
    match validate_findings_seeded(state, pr_title, &truncated, &all_findings, Some(42)).await {
        Ok(validated) => validated.into_iter().take(max_findings).collect(),
        Err(e) => {
            warn!("Validation failed: {e}");
            all_findings.into_iter().take(max_findings).collect()
        }
    }
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
