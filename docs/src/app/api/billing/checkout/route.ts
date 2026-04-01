import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import { getStripe } from "@/lib/stripe";
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
    const stripe = getStripe();
    const supabase = getSupabase();

    const { data: existing } = await supabase
      .from("credits")
      .select("stripe_customer_id")
      .eq("user_id", userId)
      .single();

    let customerId = existing?.stripe_customer_id;

    if (!customerId) {
      const customer = await stripe.customers.create({
        metadata: { clerk_user_id: userId },
      });
      customerId = customer.id;

      await supabase.from("credits").upsert({
        user_id: userId,
        stripe_customer_id: customerId,
        balance_cents: 0,
      });
    }

    const session = await stripe.checkout.sessions.create({
      customer: customerId,
      mode: "payment",
      line_items: [
        {
          price_data: {
            currency: "usd",
            unit_amount: amount,
            product_data: {
              name: `$${(amount / 100).toFixed(0)} inspect credits`,
            },
          },
          quantity: 1,
        },
      ],
      metadata: { clerk_user_id: userId, credit_cents: String(amount) },
      success_url: `${req.headers.get("origin")}/dashboard/billing?credited=true`,
      cancel_url: `${req.headers.get("origin")}/dashboard/billing`,
    });

    return NextResponse.json({ url: session.url });
  } catch (e: any) {
    return NextResponse.json(
      { error: e.message || "Stripe error" },
      { status: 500 }
    );
  }
}
