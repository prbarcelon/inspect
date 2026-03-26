use crate::openai::AgentContext;
use inspect_core::types::EntityReview;
use sem_core::model::change::ChangeType;

pub const SYSTEM_REVIEW: &str = "You are a precise code reviewer. Only report real bugs you are confident about. Always respond with valid JSON.";

pub const SYSTEM_VALIDATE: &str = "You are a precise reviewer. Verify each issue against the actual diff. Only keep confirmed bugs. Always respond with valid JSON.";

// Specialized lens system prompts
pub const SYSTEM_DATA: &str = "You are a data correctness reviewer. Always respond with valid JSON.";
pub const SYSTEM_CONCURRENCY: &str = "You are a concurrency/state bug reviewer. Always respond with valid JSON.";
pub const SYSTEM_CONTRACTS: &str = "You are an API contracts reviewer. Always respond with valid JSON.";
pub const SYSTEM_SECURITY: &str = "You are a security reviewer. Always respond with valid JSON.";
pub const SYSTEM_TYPOS: &str = "You are a character-level detail reviewer. Always respond with valid JSON.";
pub const SYSTEM_RUNTIME: &str = "You are a runtime failure analyst. Always respond with valid JSON.";

pub const PROMPT_LENS_DATA: &str = r#"You are a code reviewer specializing in DATA CORRECTNESS issues.

PR Title: {pr_title}
{triage_section}
PR Diff:
{diff}

Focus ONLY on: wrong translations, wrong constants/mappings/enum values, copy-paste errors, wrong key/field references, case sensitivity in comparisons, incorrect regex.
Rules: ONLY concrete data issues. Be specific. Max 5 issues.
Respond with ONLY: {{"issues": [{{"issue": "desc", "evidence": "code"}}]}}"#;

pub const PROMPT_LENS_CONCURRENCY: &str = r#"You are a code reviewer specializing in CONCURRENCY and STATE bugs.

PR Title: {pr_title}
{triage_section}
PR Diff:
{diff}

Focus ONLY on: race conditions, missing locks/transactions, stale reads, process lifecycle bugs, cache inconsistency, feature flag inconsistency.
Rules: ONLY issues with evidence in the diff. Be specific. Max 5 issues.
Respond with ONLY: {{"issues": [{{"issue": "desc", "evidence": "code"}}]}}"#;

pub const PROMPT_LENS_CONTRACTS: &str = r#"You are a code reviewer specializing in API CONTRACT violations.

PR Title: {pr_title}
{triage_section}
PR Diff:
{diff}

Focus ONLY on: missing abstract method implementations, wrong signatures/types, API breaking changes, wrong parameter order, key mismatches, missing React keys, import errors, method name typos breaking interfaces.
Rules: ONLY verifiable issues. Be specific. Max 5 issues.
Respond with ONLY: {{"issues": [{{"issue": "desc", "evidence": "code"}}]}}"#;

pub const PROMPT_LENS_SECURITY: &str = r#"You are a security-focused code reviewer.

PR Title: {pr_title}
{triage_section}
PR Diff:
{diff}

Focus ONLY on: SSRF, XSS, injection, auth bypass, origin/referrer bypass, case sensitivity bypass in security comparisons, frame options misconfig, hardcoded secrets.
Rules: ONLY real exploitable vulnerabilities. Be specific. Max 5 issues.
Respond with ONLY: {{"issues": [{{"issue": "desc", "evidence": "code"}}]}}"#;

pub const PROMPT_LENS_TYPOS: &str = r#"You are a code reviewer with exceptional attention to character-level detail.

PR Title: {pr_title}
{triage_section}
PR Diff:
{diff}

Focus ONLY on:
- Method/function/variable name TYPOS causing runtime errors
- Wrong language in locale/translation files
- Missing required method suffixes (Rails '?', etc.)
- Case sensitivity bugs in comparisons
- Wrong vendor prefixes
- Property/key name mismatches

Rules: Character-level precision. Only if it causes runtime failure. Max 5 issues.
Respond with ONLY: {{"issues": [{{"issue": "desc", "evidence": "code"}}]}}"#;

pub const PROMPT_LENS_RUNTIME: &str = r#"You are a code reviewer focused on RUNTIME FAILURES.

PR Title: {pr_title}
{triage_section}
PR Diff:
{diff}

For each changed function/class, ask: "What would happen if I ran this code?"

Focus ONLY on:
- Null/nil/undefined dereference
- Missing abstract method implementations causing TypeError
- Unreachable code branches
- Infinite recursion without termination
- Wrong error messages
- Panic on nil in Go
- Missing React keys

Rules: RUNTIME behavior only. Only actual failures. Max 5 issues.
Respond with ONLY: {{"issues": [{{"issue": "desc", "evidence": "code"}}]}}"#;

pub const PROMPT_DEEP: &str = r#"You are a world-class code reviewer. Review this PR and find ONLY real, concrete bugs.

PR Title: {pr_title}

{triage_section}

PR Diff:
{diff}

Look specifically for these categories of issues:
1. Logic errors: wrong conditions, off-by-one, incorrect algorithms, broken control flow, inverted booleans
2. Concurrency bugs: race conditions, missing locks, unsafe shared state, deadlocks, unhandled async promises
3. Null/undefined safety: missing null checks, possible NPE, Optional.get() without isPresent(), uninitialized variables
4. Error handling: swallowed exceptions, missing error propagation, wrong error types
5. Data correctness: wrong translations, wrong constants, incorrect mappings, copy-paste errors, stale cache data
6. Security: SSRF, XSS, injection, auth bypass, exposed secrets, unsafe deserialization, origin validation bypass
7. Type mismatches: wrong return types, incompatible casts, API contract violations, schema errors
8. Breaking changes: removed public APIs without migration, changed behavior silently
9. State consistency: asymmetric cache trust, orphaned data, inconsistent updates across related fields
10. Naming/contract bugs: method name typos that break interfaces, property names that don't match expected contracts

Rules:
- ONLY report issues you are highly confident about (>90% sure)
- Be specific: name the file, function/variable, and exactly what's wrong
- Naming typos ARE bugs if they would cause a runtime error or break an API contract
- Do NOT report: style preferences, missing tests, docs, "could be improved"
- Do NOT report issues about code that was only deleted/removed
- Maximum 10 issues. Quality over quantity.

For each issue, provide it as a JSON object with "issue" (description) and "evidence" (quote the specific code lines from the diff that prove this is a bug).

Respond with ONLY a JSON object:
{{"issues": [{{"issue": "description", "evidence": "the specific code"}}]}}"#;

pub const PROMPT_VALIDATE: &str = r#"You are a senior code reviewer doing final validation. You have the PR diff and candidate issues.

PR Title: {pr_title}

PR Diff (for verification):
{diff}

Candidate Issues:
{candidates}

For each candidate, verify against the actual diff:
1. Can you find the specific code that's buggy? If yes, keep it.
2. Is this a real bug that would cause incorrect behavior in production? If yes, keep it.
3. Is this about deleted/removed code being replaced? If so, DROP it.
4. Is this speculative or theoretical ("could potentially...")? If so, DROP it.
5. Is this about style, naming conventions, or missing tests? If so, DROP it.

Return ONLY the issues that are verified real bugs with evidence in the diff.

Respond with ONLY a JSON object:
{{"issues": ["verified issue 1", "verified issue 2", ...]}}"#;

/// Smart diff truncation that deprioritizes tests, docs, configs.
pub fn truncate_diff(diff: &str, max_chars: usize) -> String {
    if diff.len() <= max_chars {
        return diff.to_string();
    }

    let parts: Vec<&str> = diff.split("diff --git ").collect();
    if parts.is_empty() {
        return diff[..max_chars].to_string();
    }

    let mut scored: Vec<(f64, &str)> = Vec::new();
    for part in &parts {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }

        let adds = part.matches("\n+").count().saturating_sub(part.matches("\n+++").count());
        let dels = part.matches("\n-").count().saturating_sub(part.matches("\n---").count());
        let mod_bonus = adds.min(dels) * 2;
        let mut score = (adds + dels + mod_bonus) as f64;

        let first_line = part.lines().next().unwrap_or("").to_lowercase();

        // Deprioritize test files
        if ["test", "spec", "mock", "__test__", "fixture"]
            .iter()
            .any(|kw| first_line.contains(kw))
        {
            score *= 0.3;
        }
        // Deprioritize docs
        if [".md", ".adoc", ".txt", ".rst", "changelog", "readme"]
            .iter()
            .any(|kw| first_line.contains(kw))
        {
            score *= 0.2;
        }
        // Deprioritize snapshots/lockfiles
        if [".snap", ".lock", "package-lock", "yarn.lock"]
            .iter()
            .any(|kw| first_line.contains(kw))
        {
            score *= 0.1;
        }
        // Deprioritize config files
        if [".json", ".yaml", ".yml", ".toml", ".xml"]
            .iter()
            .any(|kw| first_line.contains(kw))
        {
            score *= 0.5;
        }

        scored.push((score, part));
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = String::new();
    for (_, part) in &scored {
        let candidate = format!("diff --git {}", part);
        if result.len() + candidate.len() > max_chars {
            break;
        }
        result.push_str(&candidate);
    }

    if result.is_empty() {
        diff[..max_chars].to_string()
    } else {
        result
    }
}

/// Build entity-grouped triage section from entity reviews.
pub fn build_rich_triage(entities: &[EntityReview]) -> String {
    if entities.is_empty() {
        return String::new();
    }

    let mut meaningful: Vec<&EntityReview> = entities
        .iter()
        .filter(|e| {
            matches!(
                e.change_type,
                ChangeType::Modified | ChangeType::Added
            ) && e.entity_type != "chunk"
        })
        .collect();

    meaningful.sort_by(|a, b| b.risk_score.partial_cmp(&a.risk_score).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<&EntityReview> = meaningful.into_iter().take(20).collect();

    if top.is_empty() {
        return String::new();
    }

    // Group by file
    let mut by_file: std::collections::HashMap<&str, Vec<&EntityReview>> =
        std::collections::HashMap::new();
    for e in &top {
        by_file.entry(e.file_path.as_str()).or_default().push(e);
    }

    let mut file_entries: Vec<(&str, Vec<&EntityReview>)> = by_file.into_iter().collect();
    file_entries.sort_by(|a, b| {
        let a_max = a.1.iter().map(|e| e.risk_score).fold(0.0_f64, f64::max);
        let b_max = b.1.iter().map(|e| e.risk_score).fold(0.0_f64, f64::max);
        b_max.partial_cmp(&a_max).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut lines = vec!["## Entity-level triage (highest-risk changes):".to_string()];
    for (fp, ents) in &file_entries {
        lines.push(format!("\n**{}**:", fp));
        for e in ents {
            let public = if e.is_public_api { " [PUBLIC API]" } else { "" };
            lines.push(format!(
                "  - {} ({}, {:?}, {}){public}",
                e.entity_name, e.entity_type, e.change_type, e.classification,
            ));
        }
    }

    lines.join("\n")
}

/// Format the PROMPT_DEEP template with actual values.
pub fn format_deep_prompt(pr_title: &str, triage_section: &str, diff: &str) -> String {
    PROMPT_DEEP
        .replace("{pr_title}", pr_title)
        .replace("{triage_section}", triage_section)
        .replace("{diff}", diff)
}

/// Format the PROMPT_VALIDATE template with actual values.
pub fn format_validate_prompt(pr_title: &str, diff: &str, candidates: &str) -> String {
    PROMPT_VALIDATE
        .replace("{pr_title}", pr_title)
        .replace("{diff}", diff)
        .replace("{candidates}", candidates)
}

/// Format a lens prompt template with actual values.
pub fn format_lens_prompt(template: &str, pr_title: &str, triage_section: &str, diff: &str) -> String {
    template
        .replace("{pr_title}", pr_title)
        .replace("{triage_section}", triage_section)
        .replace("{diff}", diff)
}

pub const SYSTEM_AGENT_CHALLENGE: &str = r#"You are a skeptical senior code reviewer. These findings already passed one round of review, so they look plausible. Your job is to try to DISPROVE each one using your tools.

For each finding:
1. Use read_file to read the FULL function/method mentioned. Don't trust the diff alone.
2. Check if the surrounding code already handles the edge case the finding describes.
3. Use search_code to check if there's error handling, validation, or fallback logic elsewhere.
4. Use get_entity to see the full before/after code and trace the logic carefully.
5. Check if the behavior described is actually intentional (read the PR title, check related code).

ONLY keep findings where you CANNOT find any reason to doubt them after investigating.
DROP a finding if:
- The surrounding code (outside the diff) handles the case
- The behavior is intentional based on other changes in the PR
- The finding misreads the logic when you trace through the full function
- The finding is about test code that doesn't affect production
- You find evidence it's a false alarm

When done, call submit_kept_indices with the NUMBERS of findings you want to keep. Do NOT rewrite or rephrase findings. Just return their indices."#;

pub fn build_agent_challenge_prompt(ctx: &AgentContext, candidates: &[crate::openai::Finding]) -> String {
    let candidates_text: String = candidates
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut line = format!("{}. {}", i + 1, f.issue);
            if let Some(ref ev) = f.evidence {
                line.push_str(&format!("\n   Evidence: {ev}"));
            }
            if let Some(ref file) = f.file {
                line.push_str(&format!(" ({})", file));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"Try to disprove these {count} findings that passed initial review. Use your tools to investigate each one.

PR Title: {title}

{triage}

Findings to challenge:
{candidates}

For each finding, use tools to check if the bug is real. Read full source files, check callers, verify the logic. Call submit_kept_indices with the NUMBERS of findings you could NOT disprove (e.g. [1, 3, 5])."#,
        count = candidates.len(),
        title = ctx.pr_title,
        triage = ctx.triage_section,
        candidates = candidates_text,
    )
}

/// Score an entity for triage ranking (matches Python entity_triage_score).
pub fn entity_triage_score(e: &EntityReview) -> f64 {
    let mut score = e.risk_score;
    score += (e.blast_radius.min(20) as f64) * 0.02;
    if e.is_public_api {
        score += 0.4;
    }
    match e.entity_type.as_str() {
        "class" | "method" | "function" | "interface" | "constructor" | "export" => score += 0.4,
        "field" => score += 0.2,
        "property" | "chunk" | "heading" => score -= 0.3,
        _ => {}
    }
    if matches!(e.change_type, ChangeType::Modified) {
        score += 0.15;
    }
    score += (e.dependent_count.min(10) as f64) * 0.03;
    let code_len = e.after_content.as_ref()
        .or(e.before_content.as_ref())
        .map(|s| s.len())
        .unwrap_or(0);
    score += (code_len.min(3000) as f64) * 0.0001;
    score
}

/// Build triage with BEFORE/AFTER code for top entities.
/// Top 10 get code snippets (800 chars each, 15K budget). Next 15 get names only.
pub fn build_code_triage(entities: &[EntityReview]) -> String {
    if entities.is_empty() {
        return String::new();
    }

    let mut meaningful: Vec<&EntityReview> = entities
        .iter()
        .filter(|e| matches!(e.change_type, ChangeType::Modified | ChangeType::Added | ChangeType::Deleted))
        .collect();

    meaningful.sort_by(|a, b| {
        entity_triage_score(b)
            .partial_cmp(&entity_triage_score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if meaningful.is_empty() {
        return build_rich_triage(entities);
    }

    let mut lines = vec!["## High-risk entities (review these carefully):".to_string()];
    let mut chars_used = 0usize;
    let code_budget = 15_000usize;

    let code_entities = &meaningful[..meaningful.len().min(10)];
    let name_entities = if meaningful.len() > 10 {
        &meaningful[10..meaningful.len().min(25)]
    } else {
        &[]
    };

    for e in code_entities {
        let mut header = format!(
            "\n### `{}` ({}, {:?}) in {}",
            e.entity_name, e.entity_type, e.change_type, e.file_path
        );
        let mut tags = Vec::new();
        if e.is_public_api {
            tags.push("PUBLIC_API".to_string());
        }
        if e.blast_radius >= 3 {
            tags.push(format!("blast={}", e.blast_radius));
        }
        if e.dependent_count >= 2 {
            let top_deps: Vec<&str> = e.dependent_names.iter().take(3).map(|(n, _)| n.as_str()).collect();
            if !top_deps.is_empty() {
                tags.push(format!("used_by={}", top_deps.join(",")));
            } else {
                tags.push(format!("used_by={}", e.dependent_count));
            }
        }
        if !tags.is_empty() {
            header.push_str(&format!(" [{}]", tags.join(", ")));
        }
        lines.push(header);

        let before: String = e.before_content.as_ref().map(|s| s.chars().take(800).collect()).unwrap_or_default();
        let after: String = e.after_content.as_ref().map(|s| s.chars().take(800).collect()).unwrap_or_default();

        let snippet = if !before.is_empty() && !after.is_empty() {
            format!("BEFORE:\n```\n{before}\n```\nAFTER:\n```\n{after}\n```")
        } else if !after.is_empty() {
            format!("NEW CODE:\n```\n{after}\n```")
        } else if !before.is_empty() {
            format!("DELETED CODE:\n```\n{before}\n```")
        } else {
            String::new()
        };

        if !snippet.is_empty() && chars_used + snippet.len() < code_budget {
            chars_used += snippet.len();
            lines.push(snippet);
        }
    }

    if !name_entities.is_empty() {
        lines.push("\n## Other changed entities:".to_string());
        let mut by_file: std::collections::HashMap<&str, Vec<&EntityReview>> =
            std::collections::HashMap::new();
        for e in name_entities {
            by_file.entry(e.file_path.as_str()).or_default().push(e);
        }
        let mut file_entries: Vec<(&str, Vec<&EntityReview>)> = by_file.into_iter().collect();
        file_entries.sort_by(|a, b| {
            let a_max = a.1.iter().map(|e| entity_triage_score(e)).fold(0.0_f64, f64::max);
            let b_max = b.1.iter().map(|e| entity_triage_score(e)).fold(0.0_f64, f64::max);
            b_max.partial_cmp(&a_max).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (fp, mut ents) in file_entries {
            lines.push(format!("\n**{fp}**:"));
            ents.sort_by(|a, b| {
                entity_triage_score(b)
                    .partial_cmp(&entity_triage_score(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for e in ents {
                lines.push(format!("  - {} ({}, {:?})", e.entity_name, e.entity_type, e.change_type));
            }
        }
    }

    lines.join("\n")
}
