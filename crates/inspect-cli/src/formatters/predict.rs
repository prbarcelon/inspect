use colored::Colorize;
use inspect_core::types::{PredictResult, RiskLevel};

pub fn print_terminal(result: &PredictResult) {
    if result.threats.is_empty() {
        println!("{}", "No entities at risk.".dimmed());
        return;
    }

    let b = &result.at_risk_by_level;
    println!(
        "\n{}  {} entities at risk from {} changes",
        "predict".bold().cyan(),
        result.total_at_risk,
        result.total_changes,
    );
    println!(
        "  {} critical, {} high, {} medium, {} low",
        format!("{}", b.critical).red().bold(),
        format!("{}", b.high).yellow().bold(),
        format!("{}", b.medium).blue(),
        format!("{}", b.low).dimmed(),
    );

    for threat in &result.threats {
        let change_icon = match threat.change_type {
            sem_core::model::change::ChangeType::Deleted => "-".red().bold(),
            sem_core::model::change::ChangeType::Modified => "~".yellow(),
            sem_core::model::change::ChangeType::Renamed => "r".blue(),
            sem_core::model::change::ChangeType::Moved => ">".blue(),
            sem_core::model::change::ChangeType::Added => "+".green().bold(),
        };

        println!(
            "\n{} {} {} {} [{}]",
            change_icon,
            format!("{}", threat.entity_type).dimmed(),
            threat.entity_name.bold(),
            format!("({})", threat.file_path).dimmed(),
            format!("{}", threat.classification),
        );

        println!("  {}:", "AT RISK".yellow().bold());

        for (i, entity) in threat.at_risk.iter().enumerate() {
            let risk_badge = match entity.risk_level {
                RiskLevel::Critical => format!(" CRITICAL ").on_red().white().bold().to_string(),
                RiskLevel::High => format!(" HIGH ").on_yellow().black().bold().to_string(),
                RiskLevel::Medium => format!(" MEDIUM ").on_blue().white().to_string(),
                RiskLevel::Low => format!(" LOW ").dimmed().to_string(),
            };

            let mut flags = Vec::new();
            if entity.is_public_api {
                flags.push("public API");
            }
            if entity.is_cross_file {
                flags.push("cross-file");
            }
            let flag_str = if flags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", flags.join(", "))
            };

            println!(
                "  {}. {} {} {} [{}-{}] {} own deps{}",
                format!("{:>2}", i + 1).dimmed(),
                risk_badge,
                format!("{}", entity.entity_type).dimmed(),
                entity.entity_name.bold(),
                entity.start_line,
                entity.end_line,
                entity.own_dependent_count,
                flag_str.dimmed(),
            );
        }
    }

    // Timing
    let t = &result.timing;
    if t.total_ms > 0 {
        println!(
            "\n{}  {}ms total ({} files, {} entities)",
            "timing".dimmed(),
            t.total_ms,
            t.file_count,
            t.graph_entity_count,
        );
        println!(
            "  diff: {}ms  graph: {}ms  scoring: {}ms",
            t.diff_ms, t.graph_build_ms, t.scoring_ms,
        );
    }

    println!();
}

pub fn print_json(result: &PredictResult) {
    let json = serde_json::to_string_pretty(result).expect("failed to serialize");
    println!("{}", json);
}

pub fn print_markdown(result: &PredictResult) {
    if result.threats.is_empty() {
        println!("No entities at risk.");
        return;
    }

    let b = &result.at_risk_by_level;
    println!(
        "# predict: {} entities at risk from {} changes",
        result.total_at_risk, result.total_changes,
    );
    println!();
    println!(
        "**Critical:** {} | **High:** {} | **Medium:** {} | **Low:** {}",
        b.critical, b.high, b.medium, b.low,
    );

    for threat in &result.threats {
        let change = format!("{:?}", threat.change_type).to_lowercase();
        println!();
        println!(
            "## `{}` ({}) in `{}` [{}]",
            threat.entity_name, threat.entity_type, threat.file_path, change,
        );
        println!();
        println!(
            "| # | Risk | Type | Entity | Lines | Own deps | Flags |"
        );
        println!(
            "|---|------|------|--------|-------|----------|-------|"
        );

        for (i, entity) in threat.at_risk.iter().enumerate() {
            let risk = match entity.risk_level {
                RiskLevel::Critical => "CRITICAL",
                RiskLevel::High => "HIGH",
                RiskLevel::Medium => "MEDIUM",
                RiskLevel::Low => "LOW",
            };

            let mut flags = Vec::new();
            if entity.is_public_api {
                flags.push("public API");
            }
            if entity.is_cross_file {
                flags.push("cross-file");
            }

            println!(
                "| {} | {} | {} | `{}` | {}-{} | {} | {} |",
                i + 1,
                risk,
                entity.entity_type,
                entity.entity_name,
                entity.start_line,
                entity.end_line,
                entity.own_dependent_count,
                flags.join(", "),
            );
        }
    }

    // Timing
    let t = &result.timing;
    if t.total_ms > 0 {
        println!();
        println!("---");
        println!(
            "*{}ms total ({} files, {} entities) | diff: {}ms, graph: {}ms, scoring: {}ms*",
            t.total_ms, t.file_count, t.graph_entity_count, t.diff_ms, t.graph_build_ms, t.scoring_ms,
        );
    }
}
