import {
  Suggestion
} from "./chunk-IMBPSLL4.mjs";

// src/components/chat/Suggestions.tsx
import { jsx } from "react/jsx-runtime";
function Suggestions({ suggestions, onSuggestionClick }) {
  return /* @__PURE__ */ jsx("div", { className: "suggestions", children: suggestions.map((suggestion, index) => /* @__PURE__ */ jsx(
    Suggestion,
    {
      title: suggestion.title,
      message: suggestion.message,
      partial: suggestion.partial,
      className: suggestion.className,
      onClick: () => onSuggestionClick(suggestion.message)
    },
    index
  )) });
}

export {
  Suggestions
};
//# sourceMappingURL=chunk-QGSPTXOV.mjs.map