(() => {
const escapeHtml = (value) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const renderTokens = (source, pattern, classify) => {
  let html = "";
  let lastIndex = 0;

  source.replace(pattern, (token, ...args) => {
    const offset = args[args.length - 2];
    html += escapeHtml(source.slice(lastIndex, offset));
    html += `<span class="${classify(token)}">${escapeHtml(token)}</span>`;
    lastIndex = offset + token.length;
    return token;
  });

  return html + escapeHtml(source.slice(lastIndex));
};

const pythonKeywords = /^(False|None|True|and|as|class|def|elif|else|for|from|if|import|in|is|not|or|return|with)$/;
const pythonBuiltins = /^(bool|dict|float|int|len|list|print|range|set|str|tuple)$/;
const pythonPattern = /("""[\s\S]*?"""|'''[\s\S]*?'''|#.*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:False|None|True|and|as|class|def|elif|else|for|from|if|import|in|is|not|or|return|with)\b|\b(?:bool|dict|float|int|len|list|print|range|set|str|tuple)\b|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*(?=\s*\())/g;
const bashPattern = /(#.*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\$\w+|--[A-Za-z0-9][\w-]*|\b(?:bash|cd|export|git|mkdocs|python|uv)\b|\b\d+(?:\.\d+)?\b)/g;

const classifyPython = (token) => {
  if (token.startsWith("#")) return "code-comment";
  if (token.startsWith("\"") || token.startsWith("'")) return "code-string";
  if (/^\d/.test(token)) return "code-number";
  if (pythonBuiltins.test(token)) return "code-builtin";
  if (pythonKeywords.test(token)) return "code-keyword";
  return "code-function";
};

const classifyBash = (token) => {
  if (token.startsWith("#")) return "code-comment";
  if (token.startsWith("\"") || token.startsWith("'")) return "code-string";
  if (token.startsWith("$")) return "code-variable";
  if (token.startsWith("--")) return "code-flag";
  if (/^\d/.test(token)) return "code-number";
  return "code-command";
};

const fitCodeBlockHeight = (block) => {
  const pre = block.closest("pre.highlight");
  if (!pre) return;

  window.requestAnimationFrame(() => {
    const style = window.getComputedStyle(pre);
    const padding = Number.parseFloat(style.paddingTop) + Number.parseFloat(style.paddingBottom);
    pre.style.minHeight = `${block.scrollHeight + padding}px`;
  });
};

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("pre.highlight code").forEach((block) => {
    if (block.dataset.syntaxHighlighted === "true") {
      return;
    }

    const language = Array.from(block.classList).find((name) => name.startsWith("language-"));
    const source = block.textContent || "";

    if (language === "language-python") {
      block.innerHTML = renderTokens(source, pythonPattern, classifyPython);
    } else if (language === "language-bash" || language === "language-sh" || language === "language-shell") {
      block.innerHTML = renderTokens(source, bashPattern, classifyBash);
    }

    block.dataset.syntaxHighlighted = "true";
    fitCodeBlockHeight(block);
  });
});
})();