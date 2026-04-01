const STRIPE_API = "https://api.stripe.com/v1";

function getKey(): string {
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) throw new Error("Missing STRIPE_SECRET_KEY");
  return key;
}

async function stripeRequest(path: string, params?: Record<string, string>): Promise<any> {
  const res = await fetch(`${STRIPE_API}${path}`, {
    method: params ? "POST" : "GET",
    headers: {
      Authorization: `Bearer ${getKey()}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: params ? new URLSearchParams(params).toString() : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error?.message || "Stripe API error");
  return data;
}

export async function createCustomer(metadata: Record<string, string>): Promise<string> {
  const params: Record<string, string> = {};
  for (const [k, v] of Object.entries(metadata)) {
    params[`metadata[${k}]`] = v;
  }
  const customer = await stripeRequest("/customers", params);
  return customer.id;
}

export async function createCheckoutSession(opts: {
  customer: string;
  amount: number;
  productName: string;
  metadata: Record<string, string>;
  successUrl: string;
  cancelUrl: string;
}): Promise<string | null> {
  const params: Record<string, string> = {
    customer: opts.customer,
    mode: "payment",
    "line_items[0][price_data][currency]": "usd",
    "line_items[0][price_data][unit_amount]": String(opts.amount),
    "line_items[0][price_data][product_data][name]": opts.productName,
    "line_items[0][quantity]": "1",
    success_url: opts.successUrl,
    cancel_url: opts.cancelUrl,
  };
  for (const [k, v] of Object.entries(opts.metadata)) {
    params[`metadata[${k}]`] = v;
  }
  const session = await stripeRequest("/checkout/sessions", params);
  return session.url;
}
