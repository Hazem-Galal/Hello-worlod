"use client";

import { CopilotSidebar, useCopilotChatSuggestions } from "@copilotkit/react-ui";

export default function DocSearchPage() {
  useCopilotChatSuggestions(
    {
      instructions:
        "Generate 2–3 document search follow-up queries based on the last user question and assistant response. " +
        "Each suggestion should be a concise search query the user might want to try next (e.g., 'What are the key benefits?', 'Show me examples'). " +
        "Keep suggestions short and actionable.",
      minSuggestions: 2,
      maxSuggestions: 3,
    },
    []
  );

  return (
    <main className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-8">
      <div className="max-w-2xl w-full text-center">
        <h1 className="text-3xl font-bold text-slate-800 mb-2">
          DocSearch
        </h1>
        <p className="text-slate-600 mb-6">
          Search and understand your documents. Ask questions in the sidebar.
        </p>
        <div className="bg-white rounded-xl shadow-md p-6 text-left text-slate-600">
          <p className="mb-2">
            <strong>Before you start:</strong> Run the RAG ingestion backend on port 8001 and ingest your documents.
          </p>
          <p className="text-sm">
            Backend: <code className="bg-slate-100 px-1 rounded">http://localhost:8001</code>
          </p>
        </div>
      </div>
      <CopilotSidebar
        defaultOpen={true}
        labels={{
          title: "DocSearch",
          initial:
            "Ask me anything about your documents. I'll search and cite sources.\n\n" +
            "Try: \"What are the main topics?\" or \"Summarize the key points.\"",
        }}
      />
    </main>
  );
}
