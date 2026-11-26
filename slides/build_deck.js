// Builds slides/token_tax_refund.pptx (black theme, palette from the user's screenshot).
// Run: NODE_PATH=<dir with node_modules/pptxgenjs> node slides/build_deck.js <path to apply_theme.js>
// Every number comes from results/ (see the paper, paper/main.tex).
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require(process.argv[2]);

const P = {                       // palette (screenshot) + black/white
  black: "000000", panel: "161616", line: "3A3A3A", mute: "A6A6A6", white: "FFFFFF",
  blue: "0FA3B1", honeydew: "D9E5D6", vanilla: "EDDEA4", toffee: "9E6240",
  saffron: "FF9B42", blush: "F9B5AC", red: "DD1C1A",
};
const THEME = {
  name: "Token Tax Black", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: {
    dk1: P.black, lt1: P.white, dk2: P.panel, lt2: P.honeydew,
    accent1: P.blue, accent2: P.saffron, accent3: P.red, accent4: P.vanilla,
    accent5: P.blush, accent6: P.toffee, hlink: P.blue, folHlink: P.blush,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";      // 13.33 x 7.5 in
pres.title = "Refunding the Token Tax";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "DARK_TITLE", background: { color: P.black },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.6, w: 11.7, h: 1.5, fontSize: 48,
      bold: true, color: P.white, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.2, w: 11.7, h: 0.9, fontSize: 22,
      color: P.honeydew, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "WHITE_SECTION", background: { color: P.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.9, w: 11.7, h: 1.3, fontSize: 44,
      bold: true, color: P.black, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.3, w: 11.7, h: 0.8, fontSize: 22,
      color: P.toffee, valign: "top", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: "7F7F7F", align: "right" },
});
pres.defineSlideMaster({
  title: "DARK_CONTENT", background: { color: P.black },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 34,
      bold: true, color: P.white, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Refunding the Token Tax  ·  CS769", options: { x: 0.6, y: 6.95, w: 6, h: 0.3, fontSize: 10,
      color: "7F7F7F", margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: "7F7F7F", align: "right" },
});

// ---------- helpers ----------
let n = 0;
const tb = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontSize: 16, color: P.white,
  objectName: `t${++n}`, ...o });
function content(title, section) {
  const s = pres.addSlide({ masterName: "DARK_CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function stat(s, x, y, w, big, label, color) {
  tb(s, big, { x, y, w, h: 0.95, fontSize: 54, bold: true, color, fontFace: THEME.headFontFace });
  tb(s, label, { x, y: y + 1.0, w, h: 0.75, fontSize: 15, color: P.honeydew, valign: "top" });
}
function card(s, x, y, w, h, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: P.panel },
    line: { color, width: 1.5 }, objectName: `card${++n}` });
}
function arrow(s, x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, objectName: `arrow${++n}`,
    line: { color: P.mute, width: 2, endArrowType: "triangle" } });
}
// token chips: the deck's motif. kind: "start" = needs a 1B call, "cont" = continuation token
function chips(s, x, y, toks, size = 0.95) {
  let cx = x;
  toks.forEach(([t, kind]) => {
    const w = size * (0.75 + 0.25 * t.length);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y, w, h: size, rectRadius: 0.1, objectName: `chip${++n}`,
      fill: { color: kind === "start" ? P.blue : P.saffron }, line: { color: P.black, width: 0 } });
    tb(s, t, { x: cx, y, w, h: size, fontSize: Math.round(size * 30), bold: true, color: P.black,
      align: "center", valign: "middle", fontFace: "Nirmala UI" });
    cx += w + 0.12;
  });
  return cx;
}
function hbar(s, labels, values, colors, o) {
  s.addChart(pres.charts.BAR, [{ name: "speed-up", labels, values }], {
    barDir: "bar", chartColors: colors, catAxisOrientation: "maxMin",
    valAxisMinVal: o.min ?? 1.0, valAxisMaxVal: o.max ?? 2.0, valAxisHidden: true,
    valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: o.fmt ?? "0.000\"×\"",
    dataLabelColor: P.white, dataLabelFontSize: 13, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: P.white, catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt",
    catAxisLineShow: false, barGapWidthPct: 45, showLegend: false,
    plotArea: { fill: { color: P.black } }, chartArea: { fill: { color: P.black } },
    x: o.x, y: o.y, w: o.w, h: o.h, objectName: `chart${++n}`,
  });
}
const caption = (s, text, y = 6.45) => tb(s, text, { x: 0.6, y, w: 12.1, h: 0.4, fontSize: 12, color: P.mute });

// ---------- slides ----------
// 1 title
pres.addSection({ title: "Motivation" });
let s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Motivation" });
s.addText("Refunding the Token Tax", { placeholder: "title" });
s.addText("Cheap, checked drafting for speculative decoding in Hindi", { placeholder: "body" });
chips(s, 0.8, 4.7, [["प्र", "start"], ["ति", "cont"], ["योगि", "cont"], ["ता", "cont"]], 0.8);
tb(s, "CS769 course project", { x: 0.8, y: 6.3, w: 8, h: 0.4, fontSize: 16, color: P.mute });
s.addNotes("Title. One Hindi word, four tokens: the first needs the draft model, the rest are the 'tax' we try to refund.");

// 2 token tax
s = content("Hindi pays a token tax", "Motivation");
stat(s, 0.6, 1.6, 3.6, "2.67", "Llama-3 tokens per Hindi word (English: 1.24)", P.saffron);
stat(s, 4.5, 1.6, 3.6, "59%", "of the 8B's Hindi tokens only continue a word that has started", P.blush);
stat(s, 8.4, 1.6, 4.3, "83–86%", "draft-model accuracy on those continuation tokens (66% at word starts)", P.blue);
card(s, 0.6, 4.05, 12.1, 2.2, P.line);
const ex = chips(s, 1.0, 4.45, [["प्र", "start"], ["ति", "cont"], ["योगि", "cont"], ["ता", "cont"]], 0.85);
tb(s, "“pratiyogita” (competition): 4 tokens, 4 draft-model calls, each 0.32 of an 8B step",
  { x: 1.0, y: 5.5, w: 11.3, h: 0.5, fontSize: 16, color: P.honeydew });
tb(s, [{ text: "word start", options: { color: P.blue, bold: true, breakLine: true } },
       { text: "continuation", options: { color: P.saffron, bold: true } }],
  { x: ex + 0.5, y: 4.45, w: 3, h: 0.85, fontSize: 16, valign: "middle" });
s.addNotes("The draft is cheap per call but not free; most Hindi tokens are continuation pieces that the draft already predicts well.");

// 3 speculative decoding
s = content("Speculative decoding in one picture", "Motivation");
const fy = 2.0, fh = 1.9;
card(s, 0.6, fy, 3.6, fh, P.blue);
tb(s, [{ text: "1B draft", options: { bold: true, fontSize: 20, color: P.blue, breakLine: true } },
       { text: "proposes a few tokens, one call each (0.32 of an 8B step)", options: { fontSize: 16 } }],
  { x: 0.85, y: fy + 0.2, w: 3.1, h: fh - 0.4, valign: "top" });
arrow(s, 4.3, fy + fh / 2, 4.9, fy + fh / 2);
card(s, 5.0, fy, 3.6, fh, P.saffron);
tb(s, [{ text: "8B target", options: { bold: true, fontSize: 20, color: P.saffron, breakLine: true } },
       { text: "checks all of them in one forward pass", options: { fontSize: 16 } }],
  { x: 5.25, y: fy + 0.2, w: 3.1, h: fh - 0.4, valign: "top" });
arrow(s, 8.7, fy + fh / 2, 9.3, fy + fh / 2);
card(s, 9.4, fy, 3.3, fh, P.honeydew);
tb(s, [{ text: "Keep", options: { bold: true, fontSize: 20, color: P.honeydew, breakLine: true } },
       { text: "the longest correct prefix + one 8B token", options: { fontSize: 16 } }],
  { x: 9.65, y: fy + 0.2, w: 2.8, h: fh - 0.4, valign: "top" });
tb(s, [{ text: "Lossless: ", options: { bold: true, color: P.vanilla } },
       { text: "the output is exactly the 8B's own greedy output. Speed depends on tokens accepted per 8B pass and on drafting cost." }],
  { x: 0.6, y: 4.5, w: 12.1, h: 0.8, fontSize: 18 });
tb(s, "Setup: Llama-3-8B-Instruct target, Llama-3.2-1B-Instruct draft, FLORES English→Hindi (tune on dev, report on devtest), 2× T4",
  { x: 0.6, y: 5.5, w: 12.1, h: 0.7, fontSize: 15, color: P.mute });
s.addNotes("Baseline drafting stops by an entropy rule: keep drafting while the draft is confident.");

// 4 phase 1
s = content("Phase 1: linguistic stopping rules lose to entropy", "Motivation");
hbar(s, ["Entropy rule", "Entropy + morphemes", "Fixed 2 tokens", "Gold word boundaries (oracle)", "Morphemes only"],
  [1.452, 1.427, 1.357, 1.353, 1.343], [P.blue, P.toffee, P.toffee, P.toffee, P.toffee],
  { x: 0.6, y: 1.5, w: 7.6, h: 4.6, min: 1.0, max: 1.6 });
card(s, 8.6, 1.7, 4.1, 3.9, P.line);
tb(s, [{ text: "Deciding when to stop drafting with word or morpheme boundaries never beat a plain entropy rule.", options: { breakLine: true } },
       { text: " ", options: { breakLine: true } },
       { text: "So the question changed: not when to stop, but how to make drafting cheaper.", options: { color: P.vanilla, bold: true } }],
  { x: 8.9, y: 1.95, w: 3.5, h: 3.4, fontSize: 17, valign: "top" });
caption(s, "Simulated lossless speed-up over the 8B alone, FLORES devtest (1,012 sentences).");
s.addNotes("Entropy at 1.452x is the baseline for everything that follows.");

// 5 refund
pres.addSection({ title: "The refund" });
s = content("The refund sits in drafting cost", "The refund");
tb(s, "1.45×", { x: 0.6, y: 1.9, w: 4.2, h: 1.4, fontSize: 80, bold: true, color: P.honeydew, fontFace: THEME.headFontFace });
tb(s, "today: entropy drafting", { x: 0.6, y: 3.35, w: 4.2, h: 0.5, fontSize: 18, color: P.mute });
arrow(s, 4.9, 2.6, 6.9, 2.6);
tb(s, "1.91×", { x: 7.1, y: 1.9, w: 5.6, h: 1.4, fontSize: 80, bold: true, color: P.saffron, fontFace: THEME.headFontFace });
tb(s, "if continuation tokens were (almost) free", { x: 7.1, y: 3.35, w: 5.6, h: 0.5, fontSize: 18, color: P.mute });
card(s, 0.6, 4.4, 12.1, 1.7, P.line);
tb(s, [{ text: "+0.46× is on the table. ", options: { bold: true, color: P.vanilla } },
       { text: "Acceptance stays the same; only the cost of drafting continuation tokens changes. A cheap completer that goes straight to the 8B must keep ≥ 70% of the draft's correct continuations to break even." }],
  { x: 0.9, y: 4.6, w: 11.5, h: 1.3, fontSize: 17, valign: "middle" });
s.addNotes("Ceiling: continuation tokens at 5% of a draft call, with the draft's own predictions (kappa=0.05): 1.911x.");

// 6 dictionaries
s = content("Word lists alone lack context", "The refund");
hbar(s, ["Dictionary (prefix)", "+ previous word + sentence cache", "+ reference words (oracle)", "1B draft model"],
  [31.3, 34.9, 47.3, 77.2], [P.toffee, P.toffee, P.toffee, P.blue],
  { x: 0.6, y: 1.5, w: 7.8, h: 4.4, min: 0, max: 100, fmt: "0.0\"%\"" });
card(s, 8.8, 1.7, 3.9, 3.9, P.line);
tb(s, [{ text: "Completing a word from its first tokens needs the sentence, not just the spelling.", options: { breakLine: true } },
       { text: " ", options: { breakLine: true } },
       { text: "Used unchecked, a dictionary drafter reaches only 1.47× (confidence-gated).", options: { color: P.vanilla, bold: true } }],
  { x: 9.1, y: 1.95, w: 3.3, h: 3.4, fontSize: 17, valign: "top" });
caption(s, "Whole-word completion accuracy at 38.7k mid-word positions of the 8B's devtest outputs; lexicon of 72.8k Hindi words.");
s.addNotes("This motivated a model-based completer: heads on the draft's hidden state.");

// 7 section: idea 1
pres.addSection({ title: "Idea 1: heads" });
s = pres.addSlide({ masterName: "WHITE_SECTION", sectionTitle: "Idea 1: heads" });
s.addText("Idea 1: teach the draft to finish words", { placeholder: "title" });
s.addText("Word-completion heads, proposals sent to the 8B unchecked", { placeholder: "body" });
tb(s, "01", { x: 0.8, y: 1.3, w: 3, h: 1.4, fontSize: 80, bold: true, color: P.blue, fontFace: THEME.headFontFace });

// 8 heads method
s = content("Word-completion heads on the 1B", "Idea 1: heads");
const cy = 1.7;
chips(s, 0.6, cy, [["प्र", "start"]], 1.0);
tb(s, "1B call", { x: 0.6, y: cy + 1.1, w: 1.6, h: 0.4, fontSize: 14, color: P.blue, bold: true });
chips(s, 2.4, cy, [["ति", "cont"], ["योगि", "cont"], ["ता", "cont"]], 1.0);
tb(s, "3 heads on the same hidden state, ~2% of a call", { x: 2.4, y: cy + 1.1, w: 4.6, h: 0.4, fontSize: 14, color: P.saffron, bold: true });
tb(s, "The next 1B call reads all three in one forward pass, then the 8B verifies as usual.",
  { x: 0.6, y: 3.4, w: 6.4, h: 0.9, fontSize: 17, color: P.honeydew });
card(s, 7.5, 1.6, 5.2, 4.6, P.line);
tb(s, [
  { text: "V0", options: { bold: true, color: P.blue } }, { text: "  plain multi-token prediction", options: { breakLine: true } },
  { text: "V1", options: { bold: true, color: P.blue } }, { text: "  loss only inside words + a “word continues?” head", options: { breakLine: true } },
  { text: "V2", options: { bold: true, color: P.blue } }, { text: "  V1 with learned loss weights", options: { breakLine: true } },
  { text: "+ lexicon", options: { bold: true, color: P.saffron } }, { text: "  stop once the word is no longer a prefix of a real Hindi word" },
], { x: 7.8, y: 1.85, w: 4.7, h: 4.1, fontSize: 16, valign: "top", paraSpaceAfter: 10 });
tb(s, "1B frozen · trained on 60k and 180k IITB pairs · restricted to 1,406 continuation tokens",
  { x: 0.6, y: 4.6, w: 6.4, h: 0.9, fontSize: 15, color: P.mute });
s.addNotes("Heads are Medusa-1 style. Their proposals reach the 8B unchecked, so a wrong head token ends the round.");

// 9 heads results
s = content("Heads collect a fifth of the refund", "Idea 1: heads");
hbar(s, ["Entropy", "Dictionary, gated", "Heads V0", "Heads V1 (word-guided)", "Heads V2", "Heads V1 + lexicon", "Ceiling"],
  [1.452, 1.471, 1.510, 1.518, 1.517, 1.542, 1.911],
  [P.honeydew, P.toffee, P.blue, P.blue, P.blue, P.saffron, P.blush],
  { x: 0.6, y: 1.45, w: 7.8, h: 4.8, min: 1.0, max: 2.1 });
card(s, 8.8, 1.6, 3.9, 2.15, P.line);
tb(s, [{ text: "57 / 50 / 44%", options: { bold: true, fontSize: 26, color: P.blue, breakLine: true } },
       { text: "head accuracy at distance 1/2/3, vs 83–86% for the 1B", options: { fontSize: 15 } }],
  { x: 9.05, y: 1.75, w: 3.4, h: 1.85, valign: "top" });
card(s, 8.8, 4.0, 3.9, 2.15, P.line);
tb(s, [{ text: "Word knowledge pays once", options: { bold: true, fontSize: 19, color: P.saffron, breakLine: true } },
       { text: "V1 beats V0 (+0.016), but with the lexicon V0 ≈ V1. 3× more data: only +0.010.", options: { fontSize: 15 } }],
  { x: 9.05, y: 4.15, w: 3.4, h: 1.85, valign: "top" });
caption(s, "Simulated, devtest, heads trained on 180k pairs. Best unchecked system: 1.542×, +0.090 of the +0.459 available.");
s.addNotes("Heads beat every baseline but are far less accurate than the draft. On real GPUs V1+lexicon gives 1.469x vs 1.384x for Entropy.");

// 10 section: idea 2
pres.addSection({ title: "Idea 2: checked drafting" });
s = pres.addSlide({ masterName: "WHITE_SECTION", sectionTitle: "Idea 2: checked drafting" });
s.addText("Idea 2: let the draft check cheap guesses", { placeholder: "title" });
s.addText("Checked drafting: the vertical cascade of Chen et al. (2024), applied to Hindi", { placeholder: "body" });
tb(s, "02", { x: 0.8, y: 1.3, w: 3, h: 1.4, fontSize: 80, bold: true, color: P.saffron, fontFace: THEME.headFontFace });

// 11 checked mechanism
s = content("Checked drafting", "Idea 2: checked drafting");
const steps = [
  ["1", "4-gram proposes", "up to 7 next tokens from a 6 MB table (0.05 ms, CPU)", P.saffron],
  ["2", "1B checks", "reads them in the call it makes anyway; keeps the prefix it agrees with", P.blue],
  ["3", "1B continues", "adds its own next token; drops the rest from its cache", P.honeydew],
  ["4", "8B verifies", "exactly as before, so the output is unchanged", P.blush],
];
steps.forEach(([num, head, body, col], i) => {
  const x = 0.6 + i * 3.1;
  card(s, x, 1.7, 2.85, 3.3, col);
  tb(s, num, { x: x + 0.25, y: 1.85, w: 1, h: 0.8, fontSize: 40, bold: true, color: col, fontFace: THEME.headFontFace });
  tb(s, head, { x: x + 0.25, y: 2.7, w: 2.4, h: 0.5, fontSize: 19, bold: true });
  tb(s, body, { x: x + 0.25, y: 3.25, w: 2.4, h: 1.6, fontSize: 15, color: P.honeydew, valign: "top" });
  if (i < 3) arrow(s, x + 2.88, 3.35, x + 3.07, 3.35);
});
tb(s, [{ text: "Every drafted token is still the 1B's own choice, ", options: { bold: true, color: P.vanilla } },
       { text: "but tokens it confirms cost no extra call. A wrong guess costs almost nothing. No training, and no knowledge of words." }],
  { x: 0.6, y: 5.35, w: 12.1, h: 0.9, fontSize: 17 });
s.addNotes("Reading up to 8 tokens costs the same as reading one (23.3 ms per draft call either way).");

// 12 checked results
s = content("Being checked matters more than the proposer", "Idea 2: checked drafting");
hbar(s, ["Heads V1 + lexicon (unchecked)", "Heads V1, checked", "Word trie, checked", "Word trie + context, checked",
  "4-gram, checked", "4-gram + heads, checked", "Bound: draft's own word", "Ceiling"],
  [1.542, 1.572, 1.582, 1.659, 1.701, 1.708, 1.857, 1.911],
  [P.blue, P.blue, P.toffee, P.toffee, P.saffron, P.saffron, P.blush, P.blush],
  { x: 0.6, y: 1.45, w: 8.2, h: 4.8, min: 1.0, max: 2.1 });
card(s, 9.2, 1.6, 3.5, 4.5, P.line);
tb(s, [{ text: "54%", options: { bold: true, fontSize: 48, color: P.saffron, breakLine: true, fontFace: THEME.headFontFace } },
       { text: "of the refund collected by a 4-gram with no training", options: { fontSize: 16, breakLine: true } },
       { text: " ", options: { fontSize: 10, breakLine: true } },
       { text: "Heads on top add only +0.004 to +0.008", options: { fontSize: 16, color: P.vanilla, bold: true } }],
  { x: 9.45, y: 1.8, w: 3.0, h: 4.1, valign: "top" });
caption(s, "Simulated, devtest (1,012 sentences), tuned on dev. 4-gram from 190k IITB Hindi sentences.");
s.addNotes("Word tries, whose proposals are always real words, also stay below the 4-gram.");

// 13 why
s = content("Why: cheap coverage beats accuracy", "Idea 2: checked drafting");
card(s, 0.6, 1.6, 3.9, 3.0, P.saffron);
tb(s, [{ text: "4-gram, checked", options: { bold: true, fontSize: 20, color: P.saffron, breakLine: true } },
       { text: "8.8", options: { bold: true, fontSize: 48, breakLine: true, fontFace: THEME.headFontFace } },
       { text: "tokens proposed per round, 10% accepted by the 1B", options: { fontSize: 15, color: P.honeydew } }],
  { x: 0.85, y: 1.8, w: 3.4, h: 2.6, valign: "top" });
card(s, 4.75, 1.6, 3.9, 3.0, P.blue);
tb(s, [{ text: "V1 heads, checked", options: { bold: true, fontSize: 20, color: P.blue, breakLine: true } },
       { text: "2.3", options: { bold: true, fontSize: 48, breakLine: true, fontFace: THEME.headFontFace } },
       { text: "tokens proposed per round, 22% accepted by the 1B", options: { fontSize: 15, color: P.honeydew } }],
  { x: 5.0, y: 1.8, w: 3.4, h: 2.6, valign: "top" });
stat(s, 9.1, 1.75, 3.6, "77%", "of the draft calls the 4-gram saves are word continuations (59% of all tokens)", P.blush);
tb(s, [{ text: "Unchecked, a wrong guess ends the round, so accuracy and word knowledge pay. ", options: { breakLine: true } },
       { text: "Checked, a wrong guess is nearly free, so what counts is covering many correct tokens cheaply.", options: { bold: true, color: P.vanilla } }],
  { x: 0.6, y: 4.95, w: 12.1, h: 1.2, fontSize: 17 });
s.addNotes("Gain grows with a sentence's continuation share: 14.5% to 18.8% over Entropy across quintiles (Spearman 0.21).");

// 14 real GPU
pres.addSection({ title: "Real GPUs" });
s = content("It holds on real GPUs", "Real GPUs");
hbar(s, ["Entropy", "Heads V1 + lexicon", "Heads V1, checked", "4-gram, checked", "4-gram + V0, checked"],
  [1.384, 1.469, 1.505, 1.620, 1.620], [P.honeydew, P.blue, P.blue, P.saffron, P.saffron],
  { x: 0.6, y: 1.45, w: 7.8, h: 4.6, min: 1.0, max: 1.8 });
stat(s, 8.9, 1.5, 3.8, "+0.236×", "4-gram over Entropy, 95% CI [+0.218, +0.254]", P.saffron);
stat(s, 8.9, 3.3, 3.8, "≤ 1%", "simulator vs measured speed-up, every method", P.blue);
tb(s, "96% of decodes match the 8B's own output, the same as no drafting (fp16 ties)",
  { x: 8.9, y: 5.1, w: 3.8, h: 0.9, fontSize: 15, color: P.honeydew });
caption(s, "Wall-clock speed-up over the 8B alone: 2× T4, 100 devtest sentences × 3 repeats, batch size 1, thresholds tuned in simulation.");
s.addNotes("Paired: 4-gram minus V1+lexicon = +0.151 [+0.129, +0.174]. Heads on top of the 4-gram: +0.000.");

// 15 robust
s = content("Cheap to build, robust to overhead", "Real GPUs");
s.addChart(pres.charts.LINE, [{ name: "4-gram, checked", labels: ["1k", "10k", "60k", "190k"], values: [1.615, 1.668, 1.689, 1.701] }], {
  x: 0.6, y: 1.5, w: 7.4, h: 4.4, chartColors: [P.saffron], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 10,
  valAxisMinVal: 1.4, valAxisMaxVal: 1.8, valAxisMajorUnit: 0.1, valAxisLabelColor: P.mute, valAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt",
  valAxisLabelFormatCode: "0.0\"×\"", valGridLine: { color: "333333", size: 0.5 }, catGridLine: { style: "none" },
  catAxisLabelColor: P.white, catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0.000\"×\"", dataLabelColor: P.white,
  dataLabelFontSize: 13, dataLabelFontFace: "+mn-lt", showLegend: false,
  showTitle: true, title: "Simulated speed-up vs n-gram training sentences", titleColor: P.honeydew, titleFontSize: 14,
  titleFontFace: "+mn-lt", plotArea: { fill: { color: P.black } }, chartArea: { fill: { color: P.black } }, objectName: "chart-data",
});
stat(s, 8.6, 1.6, 4.1, "1.660×", "4-gram built from just 997 outputs of the 8B, no external corpus", P.blue);
stat(s, 8.6, 3.6, 4.1, "1.610×", "even if a call that reads a proposal cost 20% more than measured", P.blush);
caption(s, "Every value above stays well above the best unchecked heads (1.542×) and Entropy (1.452×).");
s.addNotes("Little data, little compute, no training.");

// 16 limitations
pres.addSection({ title: "Wrap-up" });
s = content("Limitations and next steps", "Wrap-up");
card(s, 0.6, 1.6, 5.9, 4.6, P.red);
tb(s, "Limitations", { x: 0.9, y: 1.8, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: P.red });
tb(s, [
  { text: "Real loop: 100 sentences, batch size 1, Hugging Face on 2× T4", options: { bullet: true, breakLine: true } },
  { text: "No English control yet: the token-tax link rests on within-Hindi analyses", options: { bullet: true, breakLine: true } },
  { text: "One language pair, one model family, greedy decoding", options: { bullet: true, breakLine: true } },
  { text: "Mechanism is not new (cascade drafting); our contribution is the analysis", options: { bullet: true } },
], { x: 0.9, y: 2.4, w: 5.3, h: 3.6, fontSize: 16, valign: "top", paraSpaceAfter: 10 });
card(s, 6.8, 1.6, 5.9, 4.6, P.blue);
tb(s, "Next steps", { x: 7.1, y: 1.8, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: P.blue });
tb(s, [
  { text: "English-output control on the same pipeline", options: { bullet: true, breakLine: true } },
  { text: "Better checked proposers: the draft's own word completion bounds at 1.857×", options: { bullet: true, breakLine: true } },
  { text: "Other high-tax tokenizers and scripts", options: { bullet: true, breakLine: true } },
  { text: "An optimised serving engine and a single large GPU", options: { bullet: true } },
], { x: 7.1, y: 2.4, w: 5.3, h: 3.6, fontSize: 16, valign: "top", paraSpaceAfter: 10 });

// 17 closing
s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Wrap-up" });
s.addText("Let the draft check cheap guesses", { placeholder: "title" });
s.addText("1.62× on real GPUs with no training, against 1.38× for entropy drafting, output unchanged", { placeholder: "body" });
chips(s, 0.8, 4.6, [["प्र", "start"], ["ति", "cont"], ["योगि", "cont"], ["ता", "cont"]], 0.8);
tb(s, "Thank you  ·  Questions?", { x: 0.8, y: 6.0, w: 8, h: 0.5, fontSize: 20, color: P.vanilla, bold: true });

(async () => {
  const out = path.join(__dirname, "token_tax_refund.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
