import { NextResponse } from "next/server";
import { getStripe } from "@/lib/stripe";

export async function GET() {
  try {
    const stripe = getStripe();
    const balance = await stripe.balance.retrieve();
    return NextResponse.json({ ok: true, currency: balance.available?.[0]?.currency });
  } catch (e: any) {
    return NextResponse.json({
      ok: false,
      error: e.message,
      type: e.type,
      code: e.code,
      statusCode: e.statusCode,
      keyPrefix: process.env.STRIPE_SECRET_KEY?.slice(0, 10),
    });
  }
}
