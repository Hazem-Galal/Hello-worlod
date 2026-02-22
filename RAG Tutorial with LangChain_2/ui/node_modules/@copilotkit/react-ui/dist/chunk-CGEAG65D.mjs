import {
  useDarkMode
} from "./chunk-JGMFJZMG.mjs";

// src/components/chat/PoweredByTag.tsx
import { jsx } from "react/jsx-runtime";
function PoweredByTag({ showPoweredBy = true }) {
  const isDark = useDarkMode();
  if (!showPoweredBy) {
    return null;
  }
  const poweredByStyle = {
    visibility: "visible",
    display: "block",
    position: "static",
    textAlign: "center",
    fontSize: "12px",
    padding: "3px 0",
    color: isDark ? "rgb(69, 69, 69)" : "rgb(214, 214, 214)"
  };
  return /* @__PURE__ */ jsx("div", { children: /* @__PURE__ */ jsx("p", { className: "poweredBy", style: poweredByStyle, children: "Powered by CopilotKit" }) });
}

export {
  PoweredByTag
};
//# sourceMappingURL=chunk-CGEAG65D.mjs.map