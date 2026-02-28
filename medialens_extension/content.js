// ================================
// MediaLens Content Script (FULL)
// Smart headline extraction + badge injection
// ================================

const slovakLabels = {
  clickbait: "Clickbait",
  conspiracy: "Konšpirácia",
  false_news: "Falošné správy",
  propaganda: "Propaganda",
  satire: "Satira",
  misleading: "Zavádzajúce",
  biased: "Zaujaté",
  legitimate: "Dôveryhodné"
};

const ALLOWLIST = [
  "sme.sk",
  "dennikn.sk",
  "aktuality.sk",
  "pravda.sk",
  "hnonline.sk",
  "ta3.com",
  "startitup.sk",
  "noviny.sk",
  "teraz.sk",
  "korzar.sme.sk",
  "trend.sk",
  "euractiv.sk",
  "zive.sk",
  "touchit.sk",
  "fontech.sk",
  "refresher.sk",
  "sport.sk",
  "webnoviny.sk",
  "sita.sk",
  "plus7dni.pluska.sk"
];

// ----------------
// Smart extractor
// ----------------

const BADGE_BLACKLIST = [
  "PRÉMIOVÉ ČÍTANIE",
  "PREMIOVE CITANIE",
  "PREMIUM",
  "REKLAMA",
  "INZERÁT",
  "INZERCIA",
  "SPONZOROVANÉ",
  "SLEDOVAŤ",
  "DISKUSIA",
  "VIDEO",
  "FOTO",
  "NAŽIVO",
  "LIVE",
  "PODCAST",
];

const HREF_BLACKLIST = [
  "/predplatne",
  "/predplatné",
  "/konto",
  "/prihlas",
  "/registr",
  "/subscribe",
  "/paywall",
  "#",
  "javascript:",
];

function norm(s) {
  return (s || "")
    .replace(/\s+/g, " ")
    .replace(/\u00A0/g, " ")
    .trim();
}

function looksLikeBadge(text) {
  const t = norm(text);
  if (!t) return true;

  const upper = t.toUpperCase();
  if (BADGE_BLACKLIST.some((b) => upper.includes(b))) return true;

  // tiny all-caps labels are usually badges
  const isAllCaps = t.length <= 28 && t === upper && /[A-ZÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ]/.test(t);
  if (isAllCaps) return true;

  // very short single-word labels
  const words = t.split(" ").filter(Boolean);
  if (t.length < 12 || words.length < 2) return true;

  return false;
}

function badHref(href) {
  if (!href) return true;
  const h = href.toLowerCase();
  return HREF_BLACKLIST.some((x) => h.includes(x));
}

function absolutizeUrl(href) {
  try {
    return new URL(href, window.location.href).toString();
  } catch {
    return href;
  }
}

// Get best possible "full title" even when UI shows "..."
function getBestTitleFromNode(node) {
  if (!node) return "";

  const candidates = [];

  // 1) attributes on node
  if (node.getAttribute) {
    candidates.push(node.getAttribute("title"));
    candidates.push(node.getAttribute("aria-label"));
    candidates.push(node.getAttribute("data-title"));
    candidates.push(node.getAttribute("data-full-title"));
  }

  // 2) if inside link, check link attrs
  const a = node.closest ? node.closest("a") : null;
  if (a) {
    candidates.push(a.getAttribute("title"));
    candidates.push(a.getAttribute("aria-label"));
    candidates.push(a.getAttribute("data-title"));
  }

  // 3) text content (last resort)
  candidates.push(node.textContent);

  const cleaned = candidates
    .map(norm)
    .filter(Boolean)
    .filter((t) => t !== "..." && t !== "…" && t !== "… ");

  if (!cleaned.length) return "";

  const visible = norm(node.textContent);
  const prefersAttrs = visible.endsWith("...") || visible.endsWith("…") || visible.includes("...");

  let best = "";
  for (const t of cleaned) {
    if (t.length < 10) continue;
    if (prefersAttrs && t === visible) continue;
    if (t.length > best.length) best = t;
  }

  return best || cleaned.sort((a, b) => b.length - a.length)[0];
}

function candidateSelectorsForHost(host) {
  if (/aktuality\.sk$/.test(host)) {
    return [
      "article a[href] h1, article a[href] h2, article a[href] h3",
      "article h1 a[href], article h2 a[href], article h3 a[href]",
      "a[href*='/clanok/'] h1, a[href*='/clanok/'] h2, a[href*='/clanok/'] h3",
      "a[href*='/clanok/']",
    ];
  }
  if (/sme\.sk$/.test(host) || /korzar\.sme\.sk$/.test(host)) {
    return [
      "article h1 a[href], article h2 a[href], article h3 a[href]",
      "a[href*='/c/'] h1, a[href*='/c/'] h2, a[href*='/c/'] h3",
      "a[href*='/c/']",
      "a[href*='/clanok/']",
    ];
  }
  if (/pluska\.sk$/.test(host) || /plus7dni\.pluska\.sk$/.test(host)) {
    return [
      "article h1 a[href], article h2 a[href], article h3 a[href]",
      "a[href] h2, a[href] h3",
      "a[href*='/clanok/']",
    ];
  }

  return [
    "article h1 a[href], article h2 a[href], article h3 a[href]",
    "article a[href] h1, article a[href] h2, article a[href] h3",
    "h1 a[href], h2 a[href], h3 a[href]",
    "a[href] h1, a[href] h2, a[href] h3",
    "a[href] span",
    "article a[href] span",
    "a[href] span[class*='title'], a[href] span[class*='headline']",
  ];
}

// ----------------
// Your existing logic (with fixes)
// ----------------

const seenKeys = new Set();

function normText(s) {
  return (s || "").trim().replace(/\s+/g, " ").toLowerCase();
}

function makeKey(el, text) {
  const a = el.closest ? el.closest("a") : null;
  const href = a ? (a.href || "") : "";
  return `${normText(text)}|${href}`;
}

function hasBadge(el) {
  return !!el.querySelector(":scope > .medialens-badge");
}

const host = location.hostname.replace(/^www\./, "");
const allowed = ALLOWLIST.some(d => host === d || host.endsWith("." + d));
if (!allowed) {
  console.log("MediaLens: site not allowlisted:", host);
  throw new Error("MediaLens disabled on this site");
}

function isValidHeadline(el) {
  if (!el) return false;
  if (!(el instanceof HTMLElement)) return false;
  if (el.dataset.medialensDone === "1") return false;

  const txt = (el.innerText || "").trim();

  // filter badges/labels just in case
  if (looksLikeBadge(txt)) return false;

  // min/max length sanity
  if (txt.length < 12 || txt.length > 220) return false;

  // avoid nav/header/footer etc.
  const badParents = ["nav", "footer", "header"];
  if (badParents.some(tag => el.closest(tag))) return false;

  return true;
}

function getBadgeTarget(node) {
  if (!node) return null;

  if (node.matches && node.matches("h1,h2,h3")) return node;

  const heading = node.querySelector?.("h1,h2,h3");
  if (heading) return heading;

  const a = node.closest?.("a[href]") || node;
  return a instanceof HTMLElement ? a : null;
}

function pickHeadlines(root = document, { limit = 120 } = {}) {
  const host = window.location.host.replace(/^www\./, "");
  const selectors = candidateSelectorsForHost(host);

  const out = [];
  const localSeen = new Set();

  for (const sel of selectors) {
    const nodes = Array.from(root.querySelectorAll(sel));

    for (const node of nodes) {
      const a = node.closest ? node.closest("a[href]") : null;
      if (!a) continue;

      const href = a.getAttribute("href");
      if (badHref(href)) continue;

      const title = getBestTitleFromNode(node);
      if (!title) continue;
      if (looksLikeBadge(title)) continue;

      const target = getBadgeTarget(node);
      if (!target) continue;

      if (!isValidHeadline(target)) continue;

      const url = absolutizeUrl(href);
      const key = `${title.toLowerCase()}|${url}`;
      if (localSeen.has(key)) continue;
      localSeen.add(key);

      // store full title for classification (fix for "...")
      target.dataset.medialensFullTitle = title;

      out.push(target);
      if (out.length >= limit) return out;
    }
  }

  return out;
}

function badge(label, confidence) {
  const b = document.createElement("span");
  b.className = "medialens-badge";

  const percent = Math.round((confidence || 0) * 100);

  b.innerHTML = `
    <span class="dot"></span>
    <span class="t">${label}</span>
    <span class="p">${percent}%</span>
  `;

  // 🔥 COLOR LOGIC
  if ((confidence || 0) >= 0.60) {
    b.style.color = "#3ddc84";   // zelená
  } else {
    b.style.color = "#ff5c5c";   // červená
  }

  return b;
}

function classify(text) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: "ML_CLASSIFY", text }, resolve);
  });
}

const io = new IntersectionObserver(async (entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;

    const el = e.target;

    // Use full title if available (fix for "...")
    const text = (el.dataset.medialensFullTitle || el.innerText || "").trim();
    if (!text) {
      io.unobserve(el);
      continue;
    }

    // ak už má badge, nič nerob
    if (hasBadge(el)) {
      io.unobserve(el);
      continue;
    }

    const key = makeKey(el, text);
    if (seenKeys.has(key)) {
      el.dataset.medialensDone = "1";
      io.unobserve(el);
      continue;
    }
    seenKeys.add(key);

    el.dataset.medialensDone = "1";
    io.unobserve(el);

    const resp = await classify(text);
    if (!resp?.ok || !resp.data) continue;

    const lab = slovakLabels[resp.data.label] || resp.data.label;
    const conf = typeof resp.data.confidence === "number" ? resp.data.confidence : 0;

    if (!hasBadge(el)) el.appendChild(badge(lab, conf));
  }
}, { threshold: 0.6 });

function observeAll() {
  pickHeadlines(document).forEach(h => io.observe(h));
}

observeAll();

// infinite scroll / dynamic pages
const mo = new MutationObserver((muts) => {
  for (const m of muts) {
    for (const n of m.addedNodes) {
      if (!(n instanceof HTMLElement)) continue;
      pickHeadlines(n).forEach(h => io.observe(h));
    }
  }
});
mo.observe(document.documentElement, { childList: true, subtree: true });