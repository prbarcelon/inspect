"use client";

import { useEffect, useState } from "react";

interface Transaction {
  id: string;
  amount_cents: number;
  type: string;
  tokens_used: number | null;
  created_at: string;
}

const CREDIT_OPTIONS = [
  { label: "$10", cents: 10_00 },
  { label: "$25", cents: 25_00 },
  { label: "$50", cents: 50_00 },
  { label: "$100", cents: 100_00 },
];

export default function BillingPage() {
  const [balanceCents, setBalanceCents] = useState(0);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [addingCredits, setAddingCredits] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch("/api/billing/balance").then((r) => r.json()),
      fetch("/api/billing/transactions").then((r) => r.json()),
    ])
      .then(([bal, txns]) => {
        setBalanceCents(bal.balance_cents || 0);
        setTransactions(txns.transactions || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const addCredits = async (cents: number) => {
    setAddingCredits(true);
    try {
      const res = await fetch("/api/billing/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ amount: cents }),
      });
      const data = await res.json();
      if (data.url) {
        window.location.href = data.url;
      }
    } finally {
      setAddingCredits(false);
    }
  };

  return (
    <div>
      <h1
        style={{
          fontSize: 28,
          fontWeight: 700,
          color: "var(--accent)",
          letterSpacing: "-1px",
          marginBottom: 12,
        }}
      >
        Billing
      </h1>
      <p
        style={{
          fontSize: 14,
          color: "var(--dim)",
          lineHeight: 1.7,
          marginBottom: 32,
        }}
      >
        Manage credits and view transaction history.
      </p>

      {loading ? (
        <p style={{ fontSize: 13, color: "var(--dim)" }}>Loading...</p>
      ) : (
        <>
          {/* Balance */}
          <div
            style={{
              border: `1px solid ${balanceCents > 0 ? "var(--green)" : "var(--red)"}`,
              borderRadius: 8,
              padding: 24,
              marginBottom: 32,
            }}
          >
            <div
              style={{
                fontSize: 13,
                color: "var(--dim)",
                marginBottom: 8,
              }}
            >
              Current balance
            </div>
            <div
              style={{
                fontSize: 36,
                fontWeight: 700,
                fontFamily: "var(--mono)",
                color: balanceCents > 0 ? "var(--green)" : "var(--red)",
              }}
            >
              ${(balanceCents / 100).toFixed(2)}
            </div>
            <div
              style={{
                fontSize: 12,
                color: "var(--dim)",
                marginTop: 8,
              }}
            >
              $0.50 per 1K tokens
            </div>
          </div>

          {/* Add credits */}
          <div style={{ marginBottom: 32 }}>
            <div
              style={{
                fontSize: 14,
                fontWeight: 600,
                color: "var(--accent)",
                marginBottom: 12,
              }}
            >
              Add credits
            </div>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              {CREDIT_OPTIONS.map((opt) => (
                <button
                  key={opt.cents}
                  onClick={() => addCredits(opt.cents)}
                  disabled={addingCredits}
                  style={{
                    padding: "10px 24px",
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    color: "var(--accent)",
                    fontSize: 14,
                    fontWeight: 600,
                    fontFamily: "var(--mono)",
                    cursor: addingCredits ? "default" : "pointer",
                    opacity: addingCredits ? 0.4 : 1,
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Transaction history */}
          {transactions.length > 0 && (
            <div>
              <div
                style={{
                  fontSize: 14,
                  fontWeight: 600,
                  color: "var(--accent)",
                  marginBottom: 12,
                }}
              >
                Transaction history
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Type</th>
                    <th style={{ textAlign: "right" }}>Tokens</th>
                    <th style={{ textAlign: "right" }}>Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {transactions.map((t) => (
                    <tr key={t.id}>
                      <td>
                        {new Date(t.created_at).toLocaleDateString()}
                      </td>
                      <td>
                        <span
                          style={{
                            color:
                              t.type === "topup"
                                ? "var(--green)"
                                : "var(--dim)",
                          }}
                        >
                          {t.type === "topup" ? "Top-up" : "Usage"}
                        </span>
                      </td>
                      <td style={{ textAlign: "right", color: "var(--dim)" }}>
                        {t.tokens_used
                          ? t.tokens_used.toLocaleString()
                          : ""}
                      </td>
                      <td
                        style={{
                          textAlign: "right",
                          color:
                            t.amount_cents > 0
                              ? "var(--green)"
                              : "var(--red)",
                          fontFamily: "var(--mono)",
                        }}
                      >
                        {t.amount_cents > 0 ? "+" : ""}$
                        {(Math.abs(t.amount_cents) / 100).toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
