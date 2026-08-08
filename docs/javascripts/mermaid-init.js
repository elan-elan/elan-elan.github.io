(function () {
  function initMermaid() {
    if (!window.mermaid) {
      return;
    }

    window.mermaid.initialize({
      startOnLoad: true,
      theme: "base",
      themeVariables: {
        background: "#f3f2ea",
        primaryColor: "#e8e9df",
        primaryBorderColor: "#69746e",
        primaryTextColor: "#17201f",
        lineColor: "#275f5a",
        secondaryColor: "#d8e3dd",
        secondaryBorderColor: "#69746e",
        secondaryTextColor: "#17201f",
        tertiaryColor: "#f3f2ea",
        tertiaryBorderColor: "#aeb8b1",
        tertiaryTextColor: "#49524f",
        fontFamily: "'Avenir Next', Avenir, 'Helvetica Neue', sans-serif"
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