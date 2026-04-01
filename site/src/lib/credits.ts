import { getSupabase } from "./supabase";

// Cost per 1K tokens (in cents)
const COST_PER_1K_TOKENS = 0.5; // $0.005 per 1K tokens

export async function checkBalance(userId: string): Promise<number> {
  const supabase = getSupabase();
  const { data } = await supabase
    .from("credits")
    .select("balance_cents")
    .eq("user_id", userId)
    .single();

  return data?.balance_cents || 0;
}

export async function deductCredits(
  userId: string,
  tokensUsed: number
): Promise<{ ok: boolean; charged_cents: number; remaining_cents: number }> {
  const supabase = getSupabase();

  const costCents = Math.ceil((tokensUsed / 1000) * COST_PER_1K_TOKENS);
  if (costCents <= 0) {
    const bal = await checkBalance(userId);
    return { ok: true, charged_cents: 0, remaining_cents: bal };
  }

  const { data: current } = await supabase
    .from("credits")
    .select("balance_cents")
    .eq("user_id", userId)
    .single();

  const balance = current?.balance_cents || 0;
  if (balance < costCents) {
    return { ok: false, charged_cents: 0, remaining_cents: balance };
  }

  const newBalance = balance - costCents;

  await supabase
    .from("credits")
    .update({ balance_cents: newBalance })
    .eq("user_id", userId);

  await supabase.from("credit_transactions").insert({
    user_id: userId,
    amount_cents: -costCents,
    type: "usage",
    tokens_used: tokensUsed,
  });

  return { ok: true, charged_cents: costCents, remaining_cents: newBalance };
}
