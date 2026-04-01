import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import { createCustomer, createCheckoutSession } from "@/lib/stripe";
import { getSupabase } from "@/lib/supabase";

const CREDIT_AMOUNTS = [10_00, 25_00, 50_00, 100_00];

export async function POST(req: Request) {
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: { amount?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const amount = body.amount;
  if (!amount || !CREDIT_AMOUNTS.includes(amount)) {
    return NextResponse.json(
      { error: "Invalid amount", valid: CREDIT_AMOUNTS },
      { status: 400 }
    );
  }

  try {
    const supabase = getSupabase();

    const { data: existing } = await supabase
      .from("credits")
      .select("stripe_customer_id")
      .eq("user_id", userId)
      .single();

    let customerId = existing?.stripe_customer_id;

    if (!customerId) {
      customerId = await createCustomer({ clerk_user_id: userId });

      await supabase.from("credits").upsert({
        user_id: userId,
        stripe_customer_id: customerId,
        balance_cents: 0,
      });
    }

    const origin = req.headers.get("origin") || "https://inspect.ataraxy-labs.com";

    const url = await createCheckoutSession({
      customer: customerId,
      amount,
      productName: `$${(amount / 100).toFixed(0)} inspect credits`,
      metadata: { clerk_user_id: userId, credit_cents: String(amount) },
      successUrl: `${origin}/dashboard/billing?credited=true`,
      cancelUrl: `${origin}/dashboard/billing`,
    });

    if (url) {
      return NextResponse.json({ url });
    }
    return NextResponse.json({ error: "Failed to create checkout session" }, { status: 500 });
  } catch (e: any) {
    return NextResponse.json(
      { error: e.message || "Stripe error" },
      { status: 500 }
    );
  }
}
