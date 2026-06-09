(function () {
  function initMermaid() {
    if (!window.mermaid) {
      return;
    }

    window.mermaid.initialize({
      startOnLoad: true,
      theme: "base",
      themeVariables: {
        background: "#fbfcfe",
        primaryColor: "#f7f9fc",
        primaryBorderColor: "#9db7c1",
        primaryTextColor: "#1f2a3a",
        lineColor: "#4969a9",
        secondaryColor: "#eef2f6",
        tertiaryColor: "#ffffff",
        fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
      },
      flowchart: {
        curve: "basis",
        htmlLabels: true,
        useMaxWidth: true
      },
      securityLevel: "loose"
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initMermaid, { once: true });
  } else {
    initMermaid();
  }
})();