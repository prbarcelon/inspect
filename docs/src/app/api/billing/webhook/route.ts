import { NextResponse } from "next/server";
import { getSupabase } from "@/lib/supabase";
import crypto from "crypto";

function verifySignature(payload: string, sig: string, secret: string): boolean {
  const parts = sig.split(",").reduce((acc: Record<string, string>, part) => {
    const [k, v] = part.split("=");
    acc[k] = v;
    return acc;
  }, {});
  const timestamp = parts["t"];
  const expected = parts["v1"];
  if (!timestamp || !expected) return false;
  const signed = crypto.createHmac("sha256", secret).update(`${timestamp}.${payload}`).digest("hex");
  return crypto.timingSafeEqual(Buffer.from(signed), Buffer.from(expected));
}

export async function POST(req: Request) {
  const body = await req.text();
  const sig = req.headers.get("stripe-signature");
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

  if (webhookSecret && sig) {
    if (!verifySignature(body, sig, webhookSecret)) {
      return NextResponse.json({ error: "Invalid signature" }, { status: 400 });
    }
  }

  const event = JSON.parse(body);

  if (event.type === "checkout.session.completed") {
    const session = event.data.object;
    const userId = session.metadata?.clerk_user_id;
    const creditCents = parseInt(session.metadata?.credit_cents || "0", 10);

    if (userId && creditCents > 0) {
      const supabase = getSupabase();

      const { data: current } = await supabase
        .from("credits")
        .select("balance_cents")
        .eq("user_id", userId)
        .single();

      const newBalance = (current?.balance_cents || 0) + creditCents;

      await supabase
        .from("credits")
        .upsert({
          user_id: userId,
          balance_cents: newBalance,
          stripe_customer_id: session.customer,
        });

      await supabase.from("credit_transactions").insert({
        user_id: userId,
        amount_cents: creditCents,
        type: "topup",
        stripe_session_id: session.id,
      });
    }
  }

  return NextResponse.json({ received: true });
}
