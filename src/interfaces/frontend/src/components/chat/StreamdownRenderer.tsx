/**
 * src/interfaces/frontend/src/components/chat/StreamdownRenderer.tsx
 * Bộ render Markdown + GFM Table + Code Block + KaTeX LaTeX dựa trên streamdown@2.7.1.
 * Hỗ trợ streaming 60fps an toàn, bảo vệ công thức toán và thẻ giả HTML từ LLM,
 * đồng thời bọc <pre><code> và <table> với min-height + overflow-x: auto để chống CLS.
 */

import { memo, useEffect, useMemo, useRef, type ReactNode } from "react";
import katex from "katex";
import {
  Streamdown,
  defaultComponents,
  type Components,
  type ExtraProps,
  type StreamdownProps,
  type StreamdownTranslations,
} from "streamdown";
import { escapeRegex } from "../../utils/formatters";

export interface StreamdownRendererProps {
  content: string;
  isStreaming?: boolean;
  searchQuery?: string;
  isActiveMatch?: boolean;
}

const VIETNAMESE_TRANSLATIONS: Partial<StreamdownTranslations> = {
  copyCode: "Sao chép mã",
  copied: "Đã sao chép",
  downloadFile: "Tải tệp xuống",
  copyTable: "Sao chép bảng",
  copyTableAsMarkdown: "Sao chép dạng Markdown",
  copyTableAsCsv: "Sao chép dạng CSV",
  copyTableAsTsv: "Sao chép dạng TSV",
  downloadTable: "Tải bảng xuống",
  downloadTableAsCsv: "Tải xuống CSV",
  downloadTableAsMarkdown: "Tải xuống Markdown",
  tableFormatMarkdown: "Markdown",
  tableFormatCsv: "CSV",
  tableFormatTsv: "TSV",
  viewFullscreen: "Xem toàn màn hình",
  exitFullscreen: "Thoát toàn màn hình",
  close: "Đóng",
  copyLink: "Sao chép liên kết",
  openLink: "Mở liên kết",
  openExternalLink: "Mở liên kết ngoài?",
  externalLinkWarning: "Bạn sắp truy cập một liên kết bên ngoài.",
};

const ALLOWED_TAGS = {
  "uet-math": ["display", "formula"],
};

const STREAMDOWN_REMEND_OPTIONS: StreamdownProps["remend"] = {
  katex: false,
};

function encodeMathPayload(formula: string): string {
  return encodeURIComponent(formula).replace(
    /[_*~()!'[\]]/g,
    (ch) => `%${ch.charCodeAt(0).toString(16).toUpperCase()}`
  );
}

function decodeMathPayload(encoded: string): string {
  if (!encoded) return "";
  try {
    return decodeURIComponent(encoded);
  } catch {
    return encoded;
  }
}

function makeMathTag(formula: string, isDisplay: boolean): string {
  const encoded = encodeMathPayload(formula.trim());
  return `<uet-math display="${isDisplay ? "1" : "0"}" formula="${encoded}"></uet-math>`;
}

const VALID_HTML_TAG_PATTERN =
  /^\/?(?:b|i|strong|em|code|pre|table|thead|tbody|tr|th|td|ul|ol|li|blockquote|h[1-6]|p|a|span|div|hr|sub|sup|mark|img|br|uet-math)\b/i;

const EMOJI_KEYCAP_RE = /([0-9#*])\uFE0F?\u20E3/gu;
const EMOJI_SPAN_RE =
  /[ \t]*(?:(?:\p{Extended_Pictographic}(?:\p{Emoji_Modifier})?(?:\uFE0F|\uFE0E)?(?:\u200D(?:\p{Extended_Pictographic}|\p{Emoji})(?:\p{Emoji_Modifier})?(?:\uFE0F|\uFE0E)?)*|\p{Regional_Indicator}{2})[ \t]*)+/gu;

const NOTICE_LABEL_RE =
  /^([ \t]*(?:>+[ \t]*|[-*+][ \t]+|\d+\.[ \t]+)?)(?![*_`])((?:Lưu ý quan trọng|Thông tin quan trọng|Căn cứ quy chế|Quan trọng|Lưu ý|Chú ý)\s*:)(?![*_`])([ \t]*)/gim;

const STANDALONE_NOTICE_BLOCKQUOTE_RE =
  /^[ \t]*(?:\*\*|__)?(Lưu ý quan trọng|Thông tin quan trọng|Lưu ý|Chú ý)(?:[ \t]*:(?:\*\*|__)?|(?:\*\*|__)?[ \t]*:)[ \t]*(?=\S)/gim;

const MATH_COMMAND_RE =
  /\\(?:frac|dfrac|tfrac|sum|prod|times|div|cdot|sqrt|left|right|text|leq|geq|neq|approx|le|ge|ne)\b/;

const NON_FORMULA_CODE_RE =
  /^(?:\/|https?:\/\/|~\/|\.\.?\/|[a-z0-9._-]+\/[a-z0-9._/-]+$)/i;
const SHORTCUT_RE =
  /^(?:ctrl|cmd|alt|shift|meta|opt)(?:\s*\+\s*[a-z0-9]+)+$/i;
const DATE_RE = /^\d{1,2}\/\d{1,2}\/\d{2,4}$/;
const FILE_EXT_RE =
  /\.(?:tsx?|jsx?|py|css|html|json|svg|png|jpe?g|mp3|pdf|docx?|txt|md|csv|tsv)\b/i;

function stripDecorativeEmojis(text: string): string {
  let out = text.replace(EMOJI_KEYCAP_RE, "$1");
  out = out.replace(EMOJI_SPAN_RE, (match, offset: number, fullStr: string) => {
    const prev = offset > 0 ? fullStr[offset - 1] : "\n";
    const next =
      offset + match.length < fullStr.length
        ? fullStr[offset + match.length]
        : "\n";
    if (prev === "\n" || next === "\n") return "";
    if (/[,.;:!?\)\]]/.test(next)) return "";
    if (/[\(\[\{"'“‘]/.test(prev)) return "";
    if (/^[*_~]$/.test(prev) && !/^[ \t]/.test(match)) return "";
    if (/^[*_~]$/.test(next) && !/[ \t]$/.test(match)) return "";
    if (/^[ \t]/.test(match) || /[ \t]$/.test(match)) return " ";
    return "";
  });
  return out.replace(/[\uFE0F\uFE0E\u200B\u00AD]/g, "");
}

function cleanTableCellWhitespace(text: string): string {
  return text.replace(/^([ \t]*\|.*)$/gm, (line) =>
    line.replace(/[ \t]{2,}/g, " ")
  );
}

function boldNoticeLabels(text: string): string {
  // 1. Tự động chuẩn hóa các đoạn lưu ý đứng riêng lẻ thành khối trích dẫn (> **Lưu ý quan trọng:** ...)
  //    để hiển thị đồng nhất kể cả khi LLM quên dấu '>' ở đầu dòng.
  let out = text.replace(
    STANDALONE_NOTICE_BLOCKQUOTE_RE,
    (_fullMatch, label: string) => `> **${label.trim()}:** `
  );

  // 2. In đậm nhãn lưu ý nếu chưa được bọc **...** (trong blockquote hoặc danh sách)
  out = out.replace(
    NOTICE_LABEL_RE,
    (
      fullMatch,
      prefix: string,
      label: string,
      trailingSpace: string,
      offset: number,
      full: string
    ) => {
      const afterIdx = offset + fullMatch.length;
      const hasMoreOnLine = afterIdx < full.length && full[afterIdx] !== "\n";
      const cleanLabel = label.replace(/\s+:/, ":").trim();
      return `${prefix}**${cleanLabel}**${
        hasMoreOnLine ? trailingSpace || " " : ""
      }`;
    }
  );

  return out;
}

function isWrappedInOuterParens(str: string): boolean {
  const s = str.trim();
  if (!s.startsWith("(") || !s.endsWith(")")) return false;
  let parenDepth = 0;
  let braceDepth = 0;
  for (let i = 0; i < s.length; i += 1) {
    const ch = s[i];
    if (ch === "\\") {
      i += 1;
      continue;
    }
    if (ch === "{") braceDepth += 1;
    else if (ch === "}" && braceDepth > 0) braceDepth -= 1;
    else if (braceDepth === 0) {
      if (ch === "(") parenDepth += 1;
      else if (ch === ")") parenDepth -= 1;
      if (parenDepth === 0 && i < s.length - 1) return false;
    }
  }
  return parenDepth === 0;
}

function stripOneOuterParens(str: string): string {
  const s = str.trim();
  return isWrappedInOuterParens(s) ? s.slice(1, -1).trim() : s;
}

function splitTopLevel(
  expr: string,
  kind: "rel" | "add" | "mul"
): { parts: string[]; ops: string[] } {
  const parts: string[] = [];
  const ops: string[] = [];
  let parenDepth = 0;
  let braceDepth = 0;
  let start = 0;

  for (let i = 0; i < expr.length; i += 1) {
    const ch = expr[i];
    if (ch === "\\") {
      i += 1;
      continue;
    }
    if (ch === "{") {
      braceDepth += 1;
      continue;
    }
    if (ch === "}" && braceDepth > 0) {
      braceDepth -= 1;
      continue;
    }
    if (braceDepth > 0) continue;

    if (ch === "(") {
      parenDepth += 1;
      continue;
    }
    if (ch === ")" && parenDepth > 0) {
      parenDepth -= 1;
      continue;
    }
    if (parenDepth > 0) continue;

    if (kind === "rel") {
      const two = expr.slice(i, i + 2);
      if (two === "<=" || two === ">=" || two === "!=") {
        parts.push(expr.slice(start, i).trim());
        ops.push(two);
        i += 1;
        start = i + 1;
        continue;
      }
      if (
        ch === "=" ||
        ch === "≤" ||
        ch === "≥" ||
        ch === "≠" ||
        ch === "<" ||
        ch === ">"
      ) {
        parts.push(expr.slice(start, i).trim());
        ops.push(ch);
        start = i + 1;
      }
    } else if (kind === "add") {
      if (ch === "+") {
        parts.push(expr.slice(start, i).trim());
        ops.push("+");
        start = i + 1;
      } else if (ch === "-" || ch === "–") {
        const prev = i > 0 ? expr[i - 1] : "";
        const next = i + 1 < expr.length ? expr[i + 1] : "";
        const hasSpaces = /\s/.test(prev) && /\s/.test(next);
        const adjacentParen = prev === ")" || next === "(";
        if (hasSpaces || adjacentParen) {
          parts.push(expr.slice(start, i).trim());
          ops.push("-");
          start = i + 1;
        }
      }
    } else if (kind === "mul") {
      if (ch === "*" || ch === "×" || ch === "·" || ch === "/" || ch === "÷") {
        parts.push(expr.slice(start, i).trim());
        ops.push(ch);
        start = i + 1;
      }
    }
  }

  parts.push(expr.slice(start).trim());
  return { parts, ops };
}

function stripOuterMathDelimiters(raw: string): string {
  let s = raw.trim();
  if (
    (s.startsWith("$$") && s.endsWith("$$") && s.length > 4) ||
    (s.startsWith("\\[") && s.endsWith("\\]") && s.length > 4) ||
    (s.startsWith("\\(") && s.endsWith("\\)") && s.length > 4)
  ) {
    s = s.slice(2, -2).trim();
  } else if (s.startsWith("$") && s.endsWith("$") && s.length > 2) {
    s = s.slice(1, -1).trim();
  }
  return s;
}

function escapeLatexText(text: string): string {
  return text.replace(/([%$#&])/g, "\\$1");
}

function formatFormulaLeaf(raw: string): string {
  const s = stripOuterMathDelimiters(raw);
  if (!s) return "";
  if (MATH_COMMAND_RE.test(s)) {
    return s
      .replace(/×/g, " \\times ")
      .replace(/÷/g, " \\div ")
      .replace(/≤/g, " \\le ")
      .replace(/≥/g, " \\ge ")
      .replace(/≠/g, " \\ne ")
      .replace(/\s{2,}/g, " ")
      .trim();
  }
  if (/^-?\d+(?:[.,]\d+)?$/.test(s)) {
    return s;
  }
  if (/^-?\d+(?:[.,]\d+)?%$/.test(s)) {
    return `${s.slice(0, -1)}\\%`;
  }
  if (/^[A-Za-z](?:_[A-Za-z0-9]+|\d+)?$/.test(s)) {
    return s;
  }
  return `\\text{${escapeLatexText(s)}}`;
}

function convertPlainFormulaToLatex(rawExpr: string): string {
  const s = stripOuterMathDelimiters(rawExpr);
  if (!s) return "";

  if (MATH_COMMAND_RE.test(s)) {
    return formatFormulaLeaf(s);
  }

  const rel = splitTopLevel(s, "rel");
  if (rel.ops.length > 0 && rel.parts.every(Boolean)) {
    const relMap: Record<string, string> = {
      "=": "=",
      "<=": "\\le",
      ">=": "\\ge",
      "!=": "\\ne",
      "≤": "\\le",
      "≥": "\\ge",
      "≠": "\\ne",
      "<": "<",
      ">": ">",
    };
    let out = convertPlainFormulaToLatex(rel.parts[0]);
    for (let k = 0; k < rel.ops.length; k += 1) {
      const opLatex = relMap[rel.ops[k]] ?? rel.ops[k];
      out += ` ${opLatex} ${convertPlainFormulaToLatex(rel.parts[k + 1])}`;
    }
    return out;
  }

  const add = splitTopLevel(s, "add");
  if (add.ops.length > 0 && add.parts.every(Boolean)) {
    let out = convertPlainFormulaToLatex(add.parts[0]);
    for (let k = 0; k < add.ops.length; k += 1) {
      out += ` ${add.ops[k]} ${convertPlainFormulaToLatex(add.parts[k + 1])}`;
    }
    return out;
  }

  const mul = splitTopLevel(s, "mul");
  if (mul.ops.length > 0 && mul.parts.every(Boolean)) {
    let acc = "";
    let startIdx = 0;

    if (mul.ops[0] === "/") {
      const num = convertPlainFormulaToLatex(stripOneOuterParens(mul.parts[0]));
      const den = convertPlainFormulaToLatex(stripOneOuterParens(mul.parts[1]));
      acc = `\\frac{${num}}{${den}}`;
      startIdx = 1;
    } else {
      acc = convertPlainFormulaToLatex(mul.parts[0]);
    }

    for (let k = startIdx; k < mul.ops.length; k += 1) {
      const op = mul.ops[k];
      const rhsRaw = mul.parts[k + 1];
      if (op === "/") {
        const den = convertPlainFormulaToLatex(stripOneOuterParens(rhsRaw));
        acc = `\\frac{${acc}}{${den}}`;
      } else if (op === "÷") {
        acc = `${acc} \\div ${convertPlainFormulaToLatex(rhsRaw)}`;
      } else {
        const opLatex = op === "·" ? "\\cdot" : "\\times";
        acc = `${acc} ${opLatex} ${convertPlainFormulaToLatex(rhsRaw)}`;
      }
    }
    return acc;
  }

  if (isWrappedInOuterParens(s)) {
    const inner = s.slice(1, -1).trim();
    return `\\left(${convertPlainFormulaToLatex(inner)}\\right)`;
  }

  return formatFormulaLeaf(s);
}

function isInlineCodeFormula(inner: string): boolean {
  const s = inner.trim();
  if (!s || s.includes("\n")) return false;
  if (
    NON_FORMULA_CODE_RE.test(s) ||
    SHORTCUT_RE.test(s) ||
    DATE_RE.test(s) ||
    FILE_EXT_RE.test(s) ||
    /https?:\/\//i.test(s) ||
    /\/api\//i.test(s)
  ) {
    return false;
  }
  if (MATH_COMMAND_RE.test(s)) return true;
  if (/\([^()]*[×*+\-÷][^()]*\)\s*\/\s*[^/\s][^/]*/.test(s)) return true;
  if (/\(\([^()]+\)\s*\/\s*(?:\([^()]+\)|[^()]+)\)/.test(s)) return true;
  if (
    /^(?:Điểm\b[^=]{0,50}|[A-Za-zÀ-ỹ]{1,12}(?:_[A-Za-z0-9]+)?)\s*=\s*.+(?:[\/×*+÷]|(?:\s-\s))/i.test(
      s
    )
  ) {
    return true;
  }
  if (/[×÷]/.test(s) && /[\/=+\-()\d]/.test(s)) return true;
  return false;
}

function transformProseSegment(segment: string): string {
  if (!segment) return segment;

  // 1. Gỡ emoji/biểu tượng trang trí, làm sạch khoảng trắng thừa trong ô bảng và in đậm nhãn lưu ý
  let out = stripDecorativeEmojis(segment);
  out = cleanTableCellWhitespace(out);
  out = boldNoticeLabels(out);

  // 2. Block math: $$...$$ hoặc \[...\]
  out = out.replace(
    /(?:\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\])/g,
    (match, dollarMath?: string, bracketMath?: string) => {
      const formula = (dollarMath ?? bracketMath ?? "").trim();
      if (!formula) return match;

      // NẾU NỘI DUNG CHỈ LÀ CON SỐ ĐƠN THUẦN (ví dụ: "25", "28.19", "25 điểm", "2026"):
      // Tuyệt đối KHÔNG render thành Block Math KaTeX căn giữa dòng với khoảng đệm lớn!
      // Bóc ra thành văn bản in đậm thông thường để hiển thị tự nhiên cùng dòng.
      if (/^-?\d+(?:[.,]\d+)?(?:\s*(?:điểm|tín chỉ|%|đ))?$/i.test(formula)) {
        return ` **${formula}** `;
      }

      // Kiểm tra có thực sự là biểu thức toán học không (chứa toán tử toán học, phép tính hoặc lệnh LaTeX)
      const isActualMathFormula =
        MATH_COMMAND_RE.test(formula) ||
        /[=+\-*/×÷≤≥≠<>\\_{}^]/.test(formula) ||
        /\b(?:frac|times|div|cdot|sum|sqrt)\b/i.test(formula);

      if (!isActualMathFormula && formula.length < 20) {
        return ` ${formula} `;
      }

      return `\n\n${makeMathTag(formula, true)}\n\n`;
    }
  );

  // 3. Inline math: \(...\) hoặc $...$ (hỗ trợ cả trường hợp LLM thêm khoảng trắng sau/trước $, vd "$ \text{Điểm XT} = ... $")
  out = out.replace(
    /(?:\\\(([\s\S]*?)\\\)|(?<![\w$\\])\$([^\$\n]+?)(?<!\\)\$(?![\w$]))/g,
    (
      match,
      parenMath?: string,
      dollarMath?: string,
      offset?: number,
      full?: string
    ) => {
      if (parenMath !== undefined) {
        const formula = parenMath.trim();
        if (!formula) return match;
        return makeMathTag(formula, false);
      }

      if (dollarMath === undefined) return match;
      const formula = dollarMath.trim();
      if (!formula) return match;

      const hasOuterSpace = /^\s|\s$/.test(dollarMath);
      const hasLatexOrMathOp =
        /\\[a-zA-Z]+|[={}^_×÷≤≥≠]|(?:\d\s*[+\-*/×÷]\s*\d)/.test(formula);

      // Nếu có khoảng trắng sát dấu $, chỉ coi là toán học khi chứa lệnh LaTeX hoặc toán tử rõ ràng
      if (hasOuterSpace && !hasLatexOrMathOp) {
        return match;
      }

      // Tránh nhận nhầm tiền tệ như "$100 đến $200" hoặc điểm số đơn thuần như "$25$"
      if (
        !hasOuterSpace &&
        /^-?\d+(?:[.,]\d+)?(?:\s*(?:điểm|tín chỉ|%|đ))?$/i.test(formula)
      ) {
        return `**${formula}**`;
      }

      // Nếu công thức dài đứng riêng cả dòng ngoài bảng thì nâng thành block math cho dễ đọc
      if (typeof offset === "number" && typeof full === "string") {
        const lineStart = full.lastIndexOf("\n", offset - 1) + 1;
        const nextNl = full.indexOf("\n", offset + match.length);
        const lineEnd = nextNl === -1 ? full.length : nextNl;
        const beforeOnLine = full.slice(lineStart, offset).trim();
        const afterOnLine = full.slice(offset + match.length, lineEnd).trim();
        if (
          !beforeOnLine &&
          !afterOnLine &&
          (formula.includes("\\frac") || formula.length > 48)
        ) {
          return `\n\n${makeMathTag(formula, true)}\n\n`;
        }
      }

      return makeMathTag(formula, false);
    }
  );

  // 4. Công thức dạng plain-text đứng riêng dòng ngoài backtick (vd: "Điểm xét tuyển = ((Điểm SAT × 30) / 1600) + ...")
  out = out.replace(
    /^[ \t]*((?:Điểm\b[^=\n]{0,50}=\s*[^$\n]*\/[^$\n]+)|\(\([^()\n]+\)\s*\/\s*(?:\([^()\n]+\)|[^()\n]+)\)(?:\s*[+\-×*÷]\s*[^$\n]+)?)[ \t]*$/gim,
    (match, expr: string) => {
      const trimmed = expr.trim();
      if (
        !trimmed ||
        trimmed.includes("$") ||
        trimmed.includes("<uet-math") ||
        trimmed.includes("|") ||
        /https?:\/\//i.test(trimmed) ||
        /\[[^\]]+\]\([^)]+\)/.test(trimmed)
      ) {
        return match;
      }
      if (!/(?:[×*+()]|\s\/\s)/.test(trimmed)) {
        return match;
      }
      const latex = convertPlainFormulaToLatex(trimmed);
      return `\n\n${makeMathTag(latex, true)}\n\n`;
    }
  );

  // 5. Bảo vệ các thẻ giả HTML do LLM sinh ra (vd: <GPA>, <Phòng 105-E3>) nhưng giữ autolink & thẻ hợp lệ (kể cả <uet-math>)
  out = out.replace(/<([^>\n]+)>/g, (match, inner: string) => {
    const trimmed = inner.trim();
    if (
      VALID_HTML_TAG_PATTERN.test(trimmed) ||
      /^https?:\/\//i.test(trimmed) ||
      /^mailto:/i.test(trimmed)
    ) {
      return match;
    }
    return `&lt;${inner}&gt;`;
  });

  return out;
}

/**
 * Tách và giữ nguyên fenced code blocks (``` / ~~~) và inline code (`...`),
 * chỉ xử lý LaTeX và thẻ giả HTML ở vùng văn bản prose.
 */
function preprocessMarkdownWithMath(raw: string): string {
  if (!raw) return "";

  const parts: string[] = [];
  let i = 0;
  let proseStart = 0;

  const flushProse = (end: number) => {
    if (end > proseStart) {
      parts.push(transformProseSegment(raw.slice(proseStart, end)));
    }
  };

  while (i < raw.length) {
    // Kiểm tra fenced code block ở đầu dòng
    const atLineStart = i === 0 || raw[i - 1] === "\n";
    if (atLineStart) {
      const fenceMatch = raw.slice(i).match(/^ {0,3}(`{3,}|~{3,})/);
      if (fenceMatch) {
        flushProse(i);
        const fence = fenceMatch[1];
        const fenceChar = fence[0];
        const fenceLen = fence.length;
        const headerEnd = raw.indexOf("\n", i);

        if (headerEnd === -1) {
          parts.push(raw.slice(i));
          proseStart = raw.length;
          break;
        }

        let searchPos = headerEnd + 1;
        let blockEnd = raw.length;
        const closeRegex = new RegExp(
          `^ {0,3}${fenceChar}{${fenceLen},}[ \\t]*$`
        );

        while (searchPos <= raw.length) {
          const nextNl = raw.indexOf("\n", searchPos);
          const lineEnd = nextNl === -1 ? raw.length : nextNl;
          const line = raw.slice(searchPos, lineEnd);
          if (closeRegex.test(line)) {
            blockEnd = nextNl === -1 ? raw.length : nextNl + 1;
            break;
          }
          if (nextNl === -1) {
            blockEnd = raw.length;
            break;
          }
          searchPos = nextNl + 1;
        }

        parts.push(raw.slice(i, blockEnd));
        i = blockEnd;
        proseStart = blockEnd;
        continue;
      }
    }

    // Kiểm tra inline code span (`...`)
    if (raw[i] === "`") {
      let tickCount = 0;
      while (i + tickCount < raw.length && raw[i + tickCount] === "`") {
        tickCount += 1;
      }
      const closer = "`".repeat(tickCount);
      let scan = i + tickCount;
      let foundEnd = -1;

      while (scan < raw.length) {
        const idx = raw.indexOf(closer, scan);
        if (idx === -1) break;
        // Đảm bảo số backtick khớp chính xác
        const beforeOk = idx === 0 || raw[idx - 1] !== "`";
        const afterIdx = idx + tickCount;
        const afterOk = afterIdx >= raw.length || raw[afterIdx] !== "`";
        if (beforeOk && afterOk) {
          foundEnd = afterIdx;
          break;
        }
        scan = idx + 1;
      }

      if (foundEnd !== -1) {
        flushProse(i);
        const inner = raw.slice(i + tickCount, foundEnd - tickCount);
        if (tickCount === 1 && isInlineCodeFormula(inner)) {
          const lineStart = raw.lastIndexOf("\n", i - 1) + 1;
          const nextNl = raw.indexOf("\n", foundEnd);
          const lineEnd = nextNl === -1 ? raw.length : nextNl;
          const lineBefore = raw.slice(lineStart, i);
          const lineAfter = raw.slice(foundEnd, lineEnd);
          const isStandalone =
            lineBefore.trim() === "" && lineAfter.trim() === "";
          const latex = convertPlainFormulaToLatex(inner);
          if (isStandalone) {
            const lead = i === 0 ? "" : raw[i - 1] === "\n" ? "\n" : "\n\n";
            const trail =
              foundEnd >= raw.length
                ? ""
                : raw[foundEnd] === "\n"
                ? "\n"
                : "\n\n";
            parts.push(`${lead}${makeMathTag(latex, true)}${trail}`);
          } else {
            parts.push(makeMathTag(latex, false));
          }
        } else {
          parts.push(raw.slice(i, foundEnd));
        }
        i = foundEnd;
        proseStart = foundEnd;
        continue;
      }

      i += tickCount;
      continue;
    }

    i += 1;
  }

  flushProse(raw.length);
  return parts.join("");
}

function UetKatexMath(props: Record<string, unknown> & ExtraProps) {
  const isDisplay = props.display === "1" || props.display === "true";
  const rawFormula = typeof props.formula === "string" ? props.formula : "";

  const decoded = useMemo(() => decodeMathPayload(rawFormula), [rawFormula]);

  const html = useMemo(() => {
    if (!decoded) return "";
    const normalizedFormula =
      !isDisplay &&
      /\\frac\b/.test(decoded) &&
      !/\\displaystyle\b/.test(decoded)
        ? `\\displaystyle ${decoded}`
        : decoded;
    try {
      return katex.renderToString(normalizedFormula, {
        displayMode: isDisplay,
        throwOnError: false,
        strict: "ignore",
      });
    } catch {
      return "";
    }
  }, [decoded, isDisplay]);

  if (!html) {
    return <code className="math-fallback">{decoded}</code>;
  }

  if (isDisplay) {
    if (/^-?\d+(?:[.,]\d+)?(?:\s*(?:điểm|tín chỉ|%|đ))?$/i.test(decoded.trim())) {
      return <strong className="font-bold text-[var(--text-primary)]">{decoded.trim()}</strong>;
    }
    return (
      <span
        className="katex-block-wrapper block"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return (
    <span
      className="katex-inline-wrapper inline align-baseline"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

const STREAMDOWN_COMPONENTS: Components = {
  table: ({ className, children, node: _node, ...props }) => (
    <div className="table-container">
      <table className={`md-table ${className ?? ""}`.trim()} {...props}>
        {children}
      </table>
    </div>
  ),
  th: ({ className, children, node: _node, ...props }) => (
    <th
      className={`whitespace-nowrap break-keep ${className ?? ""}`.trim()}
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ className, children, node: _node, ...props }) => (
    <td className={`break-normal ${className ?? ""}`.trim()} {...props}>
      {children}
    </td>
  ),
  pre: ({ children, ...props }) => {
    const DefaultPre = defaultComponents.pre;
    return (
      <div className="min-h-[44px] w-full overflow-x-auto">
        <DefaultPre {...props}>{children as ReactNode}</DefaultPre>
      </div>
    );
  },
  "uet-math": UetKatexMath,
};

function clearSearchMarks(container: HTMLElement): void {
  const existing = container.querySelectorAll('mark[data-uet-search-mark="true"]');
  if (existing.length === 0) return;

  const parents = new Set<Node>();
  existing.forEach((mark) => {
    const parent = mark.parentNode;
    if (!parent) return;
    parents.add(parent);
    parent.replaceChild(document.createTextNode(mark.textContent || ""), mark);
  });
  parents.forEach((p) => p.normalize());
}

function applySearchMarks(
  container: HTMLElement,
  query: string,
  isActiveMatch: boolean
): void {
  const trimmed = query.trim();
  if (!trimmed) return;

  const regex = new RegExp(`(${escapeRegex(trimmed)})`, "gi");
  const textNodes: Text[] = [];
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const val = node.nodeValue;
      if (!val || !val.trim()) return NodeFilter.FILTER_REJECT;
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      if (
        parent.closest(
          '.katex, .katex-mathml, [data-streamdown="code-block-actions"], script, style, mark[data-uet-search-mark="true"]'
        )
      ) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let current = walker.nextNode();
  while (current) {
    textNodes.push(current as Text);
    current = walker.nextNode();
  }

  const lowerQuery = trimmed.toLowerCase();
  for (const textNode of textNodes) {
    const raw = textNode.nodeValue || "";
    if (!raw.toLowerCase().includes(lowerQuery)) continue;

    const parts = raw.split(regex);
    if (parts.length <= 1) continue;

    const frag = document.createDocumentFragment();
    for (const part of parts) {
      if (!part) continue;
      if (part.toLowerCase() === lowerQuery) {
        const mark = document.createElement("mark");
        mark.setAttribute("data-uet-search-mark", "true");
        mark.className = isActiveMatch
          ? "search-highlight search-highlight-active"
          : "search-highlight";
        mark.textContent = part;
        frag.appendChild(mark);
      } else {
        frag.appendChild(document.createTextNode(part));
      }
    }
    textNode.parentNode?.replaceChild(frag, textNode);
  }
}

export const StreamdownRenderer = memo(function StreamdownRenderer({
  content,
  isStreaming = false,
  searchQuery = "",
  isActiveMatch = false,
}: StreamdownRendererProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  const processedContent = useMemo(
    () => preprocessMarkdownWithMath(content || ""),
    [content]
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container || isStreaming) return;

    clearSearchMarks(container);
    if (searchQuery && searchQuery.trim()) {
      applySearchMarks(container, searchQuery, isActiveMatch);
    }

    return () => {
      clearSearchMarks(container);
    };
  }, [processedContent, searchQuery, isActiveMatch, isStreaming]);

  return (
    <div
      ref={containerRef}
      className={`markdown-body ${isStreaming ? "streaming-active" : ""}`}
    >
      <Streamdown
        mode={isStreaming ? "streaming" : "static"}
        isAnimating={Boolean(isStreaming)}
        parseIncompleteMarkdown={true}
        caret={isStreaming ? "block" : undefined}
        controls={{ code: true, table: false }}
        tableMaxHeight={Infinity}
        translations={VIETNAMESE_TRANSLATIONS}
        remend={STREAMDOWN_REMEND_OPTIONS}
        allowedTags={ALLOWED_TAGS}
        components={STREAMDOWN_COMPONENTS}
      >
        {processedContent}
      </Streamdown>
    </div>
  );
});