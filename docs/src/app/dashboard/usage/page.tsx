"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  request_count: number;
  last_used_at: string | null;
}

interface Review {
  id: string;
  summary: {
    usage?: { input_tokens: number; output_tokens: number };
  } | null;
  api_key_id: string | null;
}

const INPUT_PRICE = 0.2; // $/M tokens
const OUTPUT_PRICE = 15; // $/M tokens

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

function calcCost(input: number, output: number): number {
  return (input / 1_000_000) * INPUT_PRICE + (output / 1_000_000) * OUTPUT_PRICE;
}

export default function UsagePage() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/api/keys").then((r) => r.json()),
      fetch("/api/reviews?limit=100").then((r) => r.json()),
    ])
      .then(([keysData, reviewsData]) => {
        setKeys(keysData.keys || []);
        setReviews(reviewsData.reviews || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const totalRequests = keys.reduce((sum, k) => sum + k.request_count, 0);

  // Aggregate tokens across all reviews
  let totalInput = 0;
  let totalOutput = 0;
  for (const r of reviews) {
    const usage = r.summary?.usage;
    if (usage) {
      totalInput += usage.input_tokens;
      totalOutput += usage.output_tokens;
    }
  }
  const totalCost = calcCost(totalInput, totalOutput);

  return (
    <div>
      <h1
        style={{
          fontSize: 22,
          fontWeight: 600,
          color: "var(--accent)",
          marginBottom: 32,
          letterSpacing: "-0.5px",
        }}
      >
        Usage
      </h1>

      {loading ? (
        <p style={{ color: "var(--dim)" }}>Loading...</p>
      ) : (
        <>
          {/* Cost summary cards */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr 1fr",
              gap: 16,
              marginBottom: 40,
            }}
          >
            <div
              style={{
                padding: 20,
                border: "1px solid var(--border)",
                borderRadius: 8,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 28,
                  fontWeight: 700,
                  color: "var(--accent)",
                  letterSpacing: "-1px",
                }}
              >
                {totalRequests}
              </div>
              <div style={{ fontSize: 12, color: "var(--dim)", marginTop: 4 }}>
                Requests
              </div>
            </div>
            <div
              style={{
                padding: 20,
                border: "1px solid var(--border)",
                borderRadius: 8,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 28,
                  fontWeight: 700,
                  color: "var(--accent)",
                  letterSpacing: "-1px",
                }}
              >
                {formatTokens(totalInput)}
              </div>
              <div style={{ fontSize: 12, color: "var(--dim)", marginTop: 4 }}>
                Input tokens
              </div>
            </div>
            <div
              style={{
                padding: 20,
                border: "1px solid var(--border)",
                borderRadius: 8,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 28,
                  fontWeight: 700,
                  color: "var(--accent)",
                  letterSpacing: "-1px",
                }}
              >
                {formatTokens(totalOutput)}
              </div>
              <div style={{ fontSize: 12, color: "var(--dim)", marginTop: 4 }}>
                Output tokens
              </div>
            </div>
            <div
              style={{
                padding: 20,
                border: "1px solid var(--border)",
                borderRadius: 8,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 28,
                  fontWeight: 700,
                  color: "var(--green)",
                  letterSpacing: "-1px",
                }}
              >
                ${totalCost.toFixed(2)}
              </div>
              <div style={{ fontSize: 12, color: "var(--dim)", marginTop: 4 }}>
                Estimated cost
              </div>
            </div>
          </div>

          {/* Pricing info */}
          <div
            style={{
              padding: "12px 16px",
              background: "var(--surface)",
              borderRadius: 6,
              fontSize: 12,
              color: "var(--dim)",
              marginBottom: 40,
            }}
          >
            Pricing: ${INPUT_PRICE.toFixed(2)}/M input tokens, ${OUTPUT_PRICE.toFixed(2)}/M output tokens (gpt-5.2)
          </div>

          <div style={{ marginBottom: 40 }}>
            <Link
              href="/dashboard/billing"
              style={{
                display: "inline-block",
                padding: "10px 24px",
                background: "var(--accent)",
                color: "var(--bg)",
                borderRadius: 6,
                fontSize: 13,
                fontWeight: 600,
                fontFamily: "var(--mono)",
                textDecoration: "none",
              }}
            >
              Add credits
            </Link>
          </div>

          {/* Per-key table */}
          {keys.length === 0 ? (
            <p style={{ color: "var(--dim)" }}>No API keys yet.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Key</th>
                  <th style={{ textAlign: "right" }}>Requests</th>
                  <th>Last Used</th>
                </tr>
              </thead>
              <tbody>
                {keys.map((k) => (
                  <tr key={k.id}>
                    <td style={{ color: "var(--accent)" }}>{k.name}</td>
                    <td style={{ color: "var(--dim)", fontSize: 11 }}>
                      {k.prefix}...
                    </td>
                    <td style={{ textAlign: "right", color: "var(--accent)" }}>
                      {k.request_count}
                    </td>
                    <td style={{ color: "var(--dim)" }}>
                      {k.last_used_at
                        ? new Date(k.last_used_at).toLocaleDateString()
                        : "Never"}
                    </td>
                  </tr>
                ))}
                <tr style={{ borderTop: "1px solid var(--border)" }}>
                  <td
                    style={{
                      color: "var(--accent)",
                      fontWeight: 600,
                    }}
                  >
                    Total
                  </td>
                  <td></td>
                  <td
                    style={{
                      textAlign: "right",
                      color: "var(--accent)",
                      fontWeight: 600,
                    }}
                  >
                    {totalRequests}
                  </td>
                  <td></td>
                </tr>
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}
