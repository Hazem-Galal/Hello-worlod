"use client";

import dynamic from "next/dynamic";

const CopilotProvider = dynamic(
  () =>
    import("./CopilotProvider").then((mod) => ({
      default: mod.CopilotProvider,
    })),
  {
    ssr: false,
    loading: () => (
      <div className="min-h-screen flex items-center justify-center bg-slate-100">
        Loading...
      </div>
    ),
  }
);

export function ClientProviders({ children }: { children: React.ReactNode }) {
  return <CopilotProvider>{children}</CopilotProvider>;
}
