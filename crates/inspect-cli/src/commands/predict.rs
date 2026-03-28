use std::path::PathBuf;

use clap::Args;
use sem_core::git::types::DiffScope;

use crate::formatters;
use crate::OutputFormat;
use inspect_core::predict::{predict_with_options, PredictOptions};
use inspect_core::types::RiskLevel;

#[derive(Args)]
pub struct PredictArgs {
    /// Commit ref or range (e.g. HEAD~1, main..feature, abc123)
    pub target: String,

    /// Output format
    #[arg(long, value_enum, default_value = "terminal")]
    pub format: OutputFormat,

    /// Minimum risk level to show
    #[arg(long)]
    pub min_risk: Option<String>,

    /// Maximum at-risk entities per change
    #[arg(long, default_value = "10")]
    pub max_per_change: usize,

    /// Repository path
    #[arg(short = 'C', long, default_value = ".")]
    pub repo: PathBuf,
}

pub fn run(args: PredictArgs) {
    let scope = parse_scope(&args.target);
    let repo = args.repo.canonicalize().unwrap_or(args.repo.clone());

    let min_risk = args
        .min_risk
        .as_deref()
        .map(parse_risk_level)
        .unwrap_or(RiskLevel::Low);

    let options = PredictOptions {
        max_at_risk_per_change: args.max_per_change,
        min_risk,
        ..PredictOptions::default()
    };

    match predict_with_options(&repo, scope, &options) {
        Ok(result) => match args.format {
            OutputFormat::Terminal => formatters::predict::print_terminal(&result),
            OutputFormat::Json => formatters::predict::print_json(&result),
            OutputFormat::Markdown => formatters::predict::print_markdown(&result),
        },
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
}

fn parse_scope(target: &str) -> DiffScope {
    if target.contains("..") {
        let parts: Vec<&str> = target.split("..").collect();
        DiffScope::Range {
            from: parts[0].to_string(),
            to: parts[1].to_string(),
        }
    } else {
        DiffScope::Commit {
            sha: target.to_string(),
        }
    }
}

fn parse_risk_level(s: &str) -> RiskLevel {
    match s.to_lowercase().as_str() {
        "critical" => RiskLevel::Critical,
        "high" => RiskLevel::High,
        "medium" => RiskLevel::Medium,
        _ => RiskLevel::Low,
    }
}
