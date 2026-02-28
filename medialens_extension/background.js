const API_URL = "http://127.0.0.1:5000/classify";

// cache aby sme neposielali rovnaký titulok furt dookola
const cache = new Map();

// ========== CONTEXT MENU (tvoje) ==========
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeHeadline",
    title: "Analyzovať titulok (MediaLens)",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId !== "analyzeHeadline") return;

  const selectedText = (info.selectionText || "").trim();
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

  if (!selectedText) {
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon.png",
      title: "MediaLens – chyba",
      message: "Najprv označte text, ktorý chcete analyzovať."
    });
    return;
  }

  classifyText(selectedText)
    .then((data) => {
      const label = slovakLabels[data.label] || data.label;
      const msg = `Kategória: ${label}\nIstota: ${(data.confidence * 100).toFixed(1)} %`;

      chrome.notifications.create({
        type: "basic",
        iconUrl: "icon.png",
        title: "MediaLens – výsledok",
        message: msg
      });
    })
    .catch(() => {
      chrome.notifications.create({
        type: "basic",
        iconUrl: "icon.png",
        title: "MediaLens – chyba",
        message: "Nepodarilo sa spojiť so službou na 127.0.0.1:5000"
      });
    });
});

// ========== AUTOMATIC (pre content script) ==========
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type !== "ML_CLASSIFY") return;

  const text = (msg.text || "").trim();
  if (!text) return sendResponse({ ok: false });

  if (cache.has(text)) {
    return sendResponse({ ok: true, data: cache.get(text), cached: true });
  }

  classifyText(text)
    .then((data) => {
      cache.set(text, data);
      sendResponse({ ok: true, data, cached: false });
    })
    .catch((e) => {
      sendResponse({ ok: false, error: String(e) });
    });

  return true; // async
});

async function classifyText(text) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  if (!res.ok) throw new Error(`API error ${res.status}`);
  return await res.json();
}