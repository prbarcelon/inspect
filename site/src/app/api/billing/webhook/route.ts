import { NextResponse } from "next/server";
import { getStripe } from "@/lib/stripe";
import { getSupabase } from "@/lib/supabase";

export async function POST(req: Request) {
  const stripe = getStripe();
  const body = await req.text();
  const sig = req.headers.get("stripe-signature");

  // If webhook secret is configured, verify signature
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

  let event;
  if (webhookSecret && sig) {
    try {
      event = stripe.webhooks.constructEvent(body, sig, webhookSecret);
    } catch {
      return NextResponse.json({ error: "Invalid signature" }, { status: 400 });
    }
  } else {
    // In dev/without webhook secret, parse directly
    event = JSON.parse(body);
  }

  if (event.type === "checkout.session.completed") {
    const session = event.data.object;
    const userId = session.metadata?.clerk_user_id;
    const creditCents = parseInt(session.metadata?.credit_cents || "0", 10);

    if (userId && creditCents > 0) {
      const supabase = getSupabase();

      // Add credits to balance
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

      // Log the transaction
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
