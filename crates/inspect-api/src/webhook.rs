use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use tracing::{error, info, warn};

use inspect_core::analyze::analyze_remote;
use inspect_core::github::{CreateReview, GitHubClient, ReviewCommentInput};
use inspect_core::noise::is_noise_file;
use inspect_core::patch::{commentable_lines, parse_patch};
use inspect_core::risk::suggest_verdict;

use crate::openai;
use crate::prompts;
use crate::state::AppState;

type HmacSha256 = Hmac<Sha256>;

// --- Webhook payload types ---

#[derive(serde::Deserialize)]
struct WebhookPayload {
    action: Option<String>,
    installation: Option<Installation>,
    pull_request: Option<PrPayload>,
    repository: Option<RepoPayload>,
    comment: Option<CommentPayload>,
    issue: Option<IssuePayload>,
}

#[derive(serde::Deserialize)]
struct Installation {
    id: u64,
}

#[derive(serde::Deserialize)]
struct PrPayload {
    number: u64,
    head: GitRef,
}

#[derive(serde::Deserialize)]
struct GitRef {
    sha: String,
}

#[derive(serde::Deserialize)]
struct RepoPayload {
    full_name: String,
}

#[derive(serde::Deserialize)]
struct CommentPayload {
    body: Option<String>,
}

#[derive(serde::Deserialize)]
struct IssuePayload {
    number: u64,
    pull_request: Option<serde_json::Value>,
}

// --- Installation token response ---

#[derive(serde::Deserialize)]
struct InstallationToken {
    token: String,
}

// --- Signature verification ---

fn verify_signature(secret: &str, body: &[u8], signature_header: &str) -> bool {
    let Some(hex_sig) = signature_header.strip_prefix("sha256=") else {
        return false;
    };
    let Ok(expected) = hex::decode(hex_sig) else {
        return false;
    };
    let Ok(mut mac) = HmacSha256::new_from_slice(secret.as_bytes()) else {
        return false;
    };
    mac.update(body);
    let computed = mac.finalize().into_bytes();
    constant_time_eq::constant_time_eq(computed.as_slice(), &expected)
}

// --- JWT generation for GitHub App auth ---

fn generate_jwt(app_id: u64, private_key_pem: &str) -> Result<String, String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("time error: {e}"))?
        .as_secs();

    let claims = serde_json::json!({
        "iat": now - 60,
        "exp": now + 600,
        "iss": app_id,
    });

    let key = jsonwebtoken::EncodingKey::from_rsa_pem(private_key_pem.as_bytes())
        .map_err(|e| format!("invalid PEM key: {e}"))?;

    let header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256);
    jsonwebtoken::encode(&header, &claims, &key).map_err(|e| format!("JWT encode failed: {e}"))
}

// --- Installation token ---

async fn get_installation_token(
    http: &reqwest::Client,
    jwt: &str,
    installation_id: u64,
) -> Result<String, String> {
    let url = format!(
        "https://api.github.com/app/installations/{installation_id}/access_tokens"
    );

    let resp = http
        .post(&url)
        .header("Authorization", format!("Bearer {jwt}"))
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "inspect/0.1")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("installation token failed {status}: {text}"));
    }

    let token: InstallationToken = resp
        .json()
        .await
        .map_err(|e| format!("parse failed: {e}"))?;

    Ok(token.token)
}

// --- Main webhook handler ---

pub async fn handle_webhook(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    // Check GitHub App is configured
    let (app_id, private_key, webhook_secret) = match (
        &state.github_app_id,
        &state.github_app_private_key,
        &state.github_webhook_secret,
    ) {
        (Some(id), Some(key), Some(secret)) => (*id, key.clone(), secret.clone()),
        _ => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "GitHub App not configured"})),
            );
        }
    };

    // Verify webhook signature
    let signature = headers
        .get("x-hub-signature-256")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if !verify_signature(&webhook_secret, &body, signature) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "invalid signature"})),
        );
    }

    // Parse event type
    let event = headers
        .get("x-github-event")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    // Parse payload
    let payload: WebhookPayload = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => {
            warn!("Failed to parse webhook payload: {e}");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid payload"})),
            );
        }
    };

    let action = payload.action.as_deref().unwrap_or("");

    match event.as_str() {
        "pull_request" if matches!(action, "opened" | "synchronize" | "reopened") => {
            let Some(pr) = &payload.pull_request else {
                return (StatusCode::OK, Json(serde_json::json!({"status": "ignored"})));
            };
            let Some(repo) = &payload.repository else {
                return (StatusCode::OK, Json(serde_json::json!({"status": "ignored"})));
            };
            let Some(install) = &payload.installation else {
                return (StatusCode::OK, Json(serde_json::json!({"status": "ignored"})));
            };

            let repo_name = repo.full_name.clone();
            let pr_number = pr.number;
            let head_sha = pr.head.sha.clone();
            let installation_id = install.id;

            info!(
                "Webhook: PR #{pr_number} {action} on {repo_name} (install: {installation_id})"
            );

            // Spawn background review
            let state = state.clone();
            tokio::spawn(async move {
                if let Err(e) = run_webhook_review(
                    &state,
                    app_id,
                    &private_key,
                    installation_id,
                    &repo_name,
                    pr_number,
                    &head_sha,
                )
                .await
                {
                    error!("Webhook review failed for {repo_name}#{pr_number}: {e}");
                }
            });

            (
                StatusCode::OK,
                Json(serde_json::json!({"status": "review_triggered"})),
            )
        }
        "issue_comment" if action == "created" => {
            let body_text = payload
                .comment
                .as_ref()
                .and_then(|c| c.body.as_deref())
                .unwrap_or("");

            if !body_text.contains("/inspect") {
                return (StatusCode::OK, Json(serde_json::json!({"status": "ignored"})));
            }

            // Only handle comments on PRs
            let Some(issue) = &payload.issue else {
                return (StatusCode::OK, Json(serde_json::json!({"status": "ignored"})));
            };
            if issue.pull_request.is_none() {
                return (StatusCode::OK, Json(serde_json::json!({"status": "ignored"})));
            }

            let Some(repo) = &payload.repository else {
                return (StatusCode::OK, Json(serde_json::json!({"status": "ignored"})));
            };
            let Some(install) = &payload.installation else {
                return (StatusCode::OK, Json(serde_json::json!({"status": "ignored"})));
            };

            let repo_name = repo.full_name.clone();
            let pr_number = issue.number;
            let installation_id = install.id;

            info!("Webhook: /inspect command on {repo_name}#{pr_number}");

            let state = state.clone();
            tokio::spawn(async move {
                // Get head SHA from PR API
                let jwt = match generate_jwt(app_id, &private_key) {
                    Ok(j) => j,
                    Err(e) => {
                        error!("JWT generation failed: {e}");
                        return;
                    }
                };
                let token = match get_installation_token(&state.http, &jwt, installation_id).await
                {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Installation token failed: {e}");
                        return;
                    }
                };
                let client = match GitHubClient::with_token(&token) {
                    Ok(c) => c,
                    Err(e) => {
                        error!("GitHub client failed: {e}");
                        return;
                    }
                };
                let pr = match client.get_pr(&repo_name, pr_number).await {
                    Ok(p) => p,
                    Err(e) => {
                        error!("Failed to fetch PR: {e}");
                        return;
                    }
                };

                if let Err(e) = run_webhook_review(
                    &state,
                    app_id,
                    &private_key,
                    installation_id,
                    &repo_name,
                    pr_number,
                    &pr.head_sha,
                )
                .await
                {
                    error!("Webhook review failed for {repo_name}#{pr_number}: {e}");
                }
            });

            (
                StatusCode::OK,
                Json(serde_json::json!({"status": "review_triggered"})),
            )
        }
        _ => (StatusCode::OK, Json(serde_json::json!({"status": "ignored"}))),
    }
}

// --- Background review pipeline ---

async fn run_webhook_review(
    state: &AppState,
    app_id: u64,
    private_key: &str,
    installation_id: u64,
    repo: &str,
    pr_number: u64,
    head_sha: &str,
) -> Result<(), String> {
    let total_start = std::time::Instant::now();

    // Get installation token
    let jwt = generate_jwt(app_id, private_key)?;
    let token = get_installation_token(&state.http, &jwt, installation_id).await?;
    let client =
        GitHubClient::with_token(&token).map_err(|e| format!("GitHub client failed: {e}"))?;

    // Fetch PR metadata
    let pr = client
        .get_pr(repo, pr_number)
        .await
        .map_err(|e| format!("Failed to fetch PR: {e}"))?;

    info!(
        "Reviewing PR #{}: {} ({} files, +{}/-{})",
        pr.number, pr.title, pr.changed_files, pr.additions, pr.deletions
    );

    // Step 1: Triage
    let visible_files: Vec<_> = pr
        .files
        .iter()
        .filter(|f| !is_noise_file(&f.filename))
        .cloned()
        .collect();

    let file_pairs = client
        .get_file_pairs(repo, &visible_files, &pr.base_sha, &pr.head_sha)
        .await;

    let result =
        analyze_remote(&file_pairs).map_err(|e| format!("Analysis failed: {e}"))?;

    let verdict = suggest_verdict(&result);
    info!(
        "Triage: {} entities, verdict: {}",
        result.stats.total_entities, verdict
    );

    // Step 2: Fetch diff for LLM review
    let diff = fetch_diff_with_token(&state.http, &token, repo, pr_number).await?;
    let triage_section = prompts::build_code_triage(&result.entity_reviews);

    // Step 3: LLM review (v20: 9 lenses + blind validation + agentic challenge)
    let ctx = openai::AgentContext {
        entity_reviews: result.entity_reviews.clone(),
        repo: repo.to_string(),
        base_sha: pr.base_sha.clone(),
        head_sha: pr.head_sha.clone(),
        pr_title: pr.title.clone(),
        diff: diff.clone(),
        triage_section: triage_section.clone(),
    };
    let (findings, agent_iters, agent_calls) =
        openai::review_hybrid_v20(state, &pr.title, &diff, &triage_section, 7, &ctx).await;

    info!(
        "Review complete: {} findings ({} agent iters, {} tool calls) in {}ms",
        findings.len(),
        agent_iters,
        agent_calls,
        total_start.elapsed().as_millis()
    );

    // Step 4: Build review comments with precise line mapping + suggestions
    let pr_with_patches = client
        .get_pr_with_patches(repo, pr_number)
        .await
        .map_err(|e| format!("Failed to fetch patches: {e}"))?;

    let mut inline_comments: Vec<ReviewCommentInput> = Vec::new();

    for finding in &findings {
        if let Some(ref file_path) = finding.file {
            // Find the matching PR file
            let pr_file = pr_with_patches
                .files
                .iter()
                .find(|f| f.filename == *file_path || f.filename.ends_with(file_path));

            let Some(pr_file) = pr_file else { continue };
            let Some(ref patch) = pr_file.patch else { continue };

            let hunks = parse_patch(patch);
            let valid_lines = commentable_lines(&hunks);
            if valid_lines.is_empty() {
                continue;
            }

            // Determine the comment line range
            let (start_line, end_line) = if let (Some(sl), Some(el)) =
                (finding.start_line, finding.end_line)
            {
                // Clamp to valid commentable lines
                let clamped_start = valid_lines
                    .iter()
                    .find(|&&l| l >= sl)
                    .copied()
                    .unwrap_or(valid_lines[valid_lines.len() - 1]);
                let clamped_end = valid_lines
                    .iter()
                    .rev()
                    .find(|&&l| l <= el)
                    .copied()
                    .unwrap_or(clamped_start);
                (clamped_start, clamped_end.max(clamped_start))
            } else {
                // Fallback: first commentable line
                (valid_lines[0], valid_lines[0])
            };

            // Build comment body
            let severity_badge = match finding.severity.as_deref() {
                Some("critical") => "**[Critical]**",
                Some("high") => "**[High]**",
                Some("medium") => "[Medium]",
                _ => "[Low]",
            };

            let mut body = format!("{} {}", severity_badge, finding.issue);
            if let Some(ref evidence) = finding.evidence {
                body.push_str(&format!("\n\n> {}", evidence));
            }

            // Add suggestion block if the LLM provided a fix
            if let Some(ref suggestion) = finding.suggestion {
                body.push_str(&format!(
                    "\n\n```suggestion\n{}\n```",
                    suggestion
                ));
            }

            inline_comments.push(ReviewCommentInput {
                path: pr_file.filename.clone(),
                line: end_line,
                body,
                start_line: if start_line < end_line {
                    Some(start_line)
                } else {
                    None
                },
            });
        }
    }

    // Step 5: Build summary body
    let stats = &result.stats;
    let summary = format!(
        "## inspect review\n\n\
         **Triage:** {} entities analyzed | {} critical, {} high, {} medium, {} low\n\
         **Verdict:** {}\n\n\
         ### Findings ({})\n\n{}\n\n\
         ---\n\
         *Reviewed by [inspect](https://inspect.ataraxy-labs.com) | \
         Entity-level triage found {} high-risk changes*",
        stats.total_entities,
        stats.by_risk.critical,
        stats.by_risk.high,
        stats.by_risk.medium,
        stats.by_risk.low,
        verdict,
        findings.len(),
        findings
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let sev = f.severity.as_deref().unwrap_or("low");
                let file_ref = f
                    .file
                    .as_deref()
                    .map(|p| format!(" in `{p}`"))
                    .unwrap_or_default();
                format!("{}. **[{}]** {}{}", i + 1, sev, f.issue, file_ref)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        stats.by_risk.critical + stats.by_risk.high,
    );

    // Step 6: Post review
    let event = if findings.is_empty() {
        "COMMENT"
    } else {
        "COMMENT"
    };

    let review = CreateReview {
        commit_id: head_sha.to_string(),
        event: event.to_string(),
        body: summary,
        comments: inline_comments,
    };

    match client.create_review(repo, pr_number, &review).await {
        Ok(resp) => {
            info!("Posted review: {}", resp.html_url);
        }
        Err(e) => {
            error!("Failed to post review: {e}");
            return Err(format!("Failed to post review: {e}"));
        }
    }

    Ok(())
}

// --- Helpers ---

async fn fetch_diff_with_token(
    http: &reqwest::Client,
    token: &str,
    repo: &str,
    pr_number: u64,
) -> Result<String, String> {
    let url = format!("https://api.github.com/repos/{repo}/pulls/{pr_number}");
    let resp = http
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/vnd.github.v3.diff")
        .header("User-Agent", "inspect/0.1")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .await
        .map_err(|e| format!("diff request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("diff fetch failed {status}: {text}"));
    }

    resp.text().await.map_err(|e| format!("diff read failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_signature_valid() {
        let secret = "test-secret";
        let body = b"hello world";

        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body);
        let sig = hex::encode(mac.finalize().into_bytes());
        let header = format!("sha256={sig}");

        assert!(verify_signature(secret, body, &header));
    }

    #[test]
    fn test_verify_signature_invalid() {
        assert!(!verify_signature("secret", b"body", "sha256=deadbeef"));
    }

    #[test]
    fn test_verify_signature_bad_prefix() {
        assert!(!verify_signature("secret", b"body", "sha1=abc"));
    }
}
