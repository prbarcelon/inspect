import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import { getSupabase } from "@/lib/supabase";

const SIGNUP_BONUS_CENTS = 5_00; // $5 free credits

export async function GET() {
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = getSupabase();
  const { data } = await supabase
    .from("credits")
    .select("balance_cents")
    .eq("user_id", userId)
    .single();

  // New user: grant $5 signup bonus
  if (!data) {
    await supabase.from("credits").insert({
      user_id: userId,
      balance_cents: SIGNUP_BONUS_CENTS,
    });

    await supabase.from("credit_transactions").insert({
      user_id: userId,
      amount_cents: SIGNUP_BONUS_CENTS,
      type: "signup_bonus",
    });

    return NextResponse.json({ balance_cents: SIGNUP_BONUS_CENTS });
  }

  return NextResponse.json({
    balance_cents: data.balance_cents || 0,
  });
}
