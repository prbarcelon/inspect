import { NextRequest, NextResponse } from "next/server";
import { validateApiKey } from "@/lib/validate-key";
import { checkBalance, deductCredits } from "@/lib/credits";

export const maxDuration = 300;

const POLL_INTERVAL = 5000;
const MAX_POLLS = 55; // ~275s max, under Vercel's 300s limit

export async function POST(req: NextRequest) {
  const keyResult = await validateApiKey(req);
  if (!keyResult.valid) return keyResult.response;

  const balance = await checkBalance(keyResult.userId);
  if (balance <= 0) {
    return NextResponse.json(
      { error: "Insufficient credits. Add credits at https://inspect.ataraxy-labs.com/dashboard" },
      { status: 402 }
    );
  }

  const inspectApiUrl = (process.env.INSPECT_API_URL || "").replace(/"/g, "");
  const inspectApiKey = (process.env.INSPECT_API_KEY || "").replace(/"/g, "");

  if (!inspectApiUrl || !inspectApiKey) {
    return NextResponse.json(
      { error: "Server missing INSPECT_API_URL or INSPECT_API_KEY" },
      { status: 500 }
    );
  }

  let body: { repo?: string; pr_number?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { repo, pr_number } = body;
  if (!repo || !pr_number) {
    return NextResponse.json(
      { error: 'Required fields: "repo" (owner/repo), "pr_number" (integer)' },
      { status: 400 }
    );
  }

  try {
    // Submit review job to Fly.io v20 API
    const submitRes = await fetch(`${inspectApiUrl}/v1/review`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${inspectApiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ repo, pr_number }),
    });

    if (!submitRes.ok) {
      const text = await submitRes.text();
      return NextResponse.json(
        { error: `Upstream error: ${text}` },
        { status: submitRes.status }
      );
    }

    const { id: jobId } = await submitRes.json();

    // Poll until complete
    for (let i = 0; i < MAX_POLLS; i++) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL));

      const pollRes = await fetch(`${inspectApiUrl}/v1/review/${jobId}`, {
        headers: { Authorization: `Bearer ${inspectApiKey}` },
      });

      if (!pollRes.ok) continue;

      const data = await pollRes.json();
      if (data.status === "complete") {
        // Deduct credits based on tokens used
        const tokensUsed = data.result?.tokens_used || 5000;
        const deduction = await deductCredits(keyResult.userId, tokensUsed);
        return NextResponse.json({
          ...data.result,
          billing: {
            tokens_used: tokensUsed,
            charged_cents: deduction.charged_cents,
            remaining_cents: deduction.remaining_cents,
          },
        });
      } else if (data.status === "failed") {
        return NextResponse.json(
          { error: data.error || "Review failed upstream" },
          { status: 500 }
        );
      }
    }

    return NextResponse.json({ error: "Review timed out" }, { status: 504 });
  } catch (e: any) {
    return NextResponse.json(
      { error: e.message || "Review failed" },
      { status: 500 }
    );
  }
}
