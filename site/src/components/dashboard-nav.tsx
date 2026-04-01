"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const tabs = [
  { href: "/dashboard", label: "Overview" },
  { href: "/dashboard/keys", label: "Keys" },
  { href: "/dashboard/usage", label: "Usage" },
  { href: "/dashboard/billing", label: "Billing" },
];

export default function DashboardNav() {
  const pathname = usePathname();

  return (
    <div
      style={{
        display: "flex",
        gap: 24,
        borderBottom: "1px solid var(--border)",
        marginBottom: 32,
        fontSize: 13,
      }}
    >
      {tabs.map((t) => {
        const active = t.href === "/dashboard"
          ? pathname === "/dashboard"
          : pathname.startsWith(t.href);
        return (
          <Link
            key={t.href}
            href={t.href}
            style={{
              padding: "10px 0",
              color: active ? "var(--accent)" : "var(--dim)",
              textDecoration: "none",
              borderBottom: active ? "1px solid var(--accent)" : "1px solid transparent",
              marginBottom: -1,
            }}
          >
            {t.label}
          </Link>
        );
      })}
    </div>
  );
}
