"use client";

import { CopilotKit } from "@copilotkit/react-core";

export function CopilotProvider({ children }: { children: React.ReactNode }) {
  return (
    <CopilotKit
      agent="docsearch"
      runtimeUrl="/api/copilotkit"
      showDevConsole={false}
    >
      {children}
    </CopilotKit>
  );
}
