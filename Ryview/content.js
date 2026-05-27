console.log("RyView content script loaded.");

let lastCall = 0;
let activePopup = null;
let lastSelectedText = "";

function predictReview(text) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(
      { action: "PREDICT_REVIEW", text },
      (response) => {
        const error = chrome.runtime.lastError;
        if (error) {
          reject(new Error(error.message));
          return;
        }

        if (!response?.ok) {
          reject(new Error(response?.error || "Unknown API error"));
          return;
        }

        resolve(response.data);
      }
    );
  });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.action !== "GET_SELECTED_TEXT") {
    return;
  }

  sendResponse({ text: window.getSelection().toString().trim() });
});

function getSelectedText() {
  return window.getSelection().toString().trim();
}

function getSelectionRect() {
  const selection = window.getSelection();

  if (!selection.rangeCount) {
    return null;
  }

  return selection.getRangeAt(0).getBoundingClientRect();
}

async function processSelection() {
  const now = Date.now();
  if (now - lastCall < 2000) return;

  const text = getSelectedText();
  if (!text) {
    lastSelectedText = "";
    return;
  }

  if (text === lastSelectedText) return;
  lastSelectedText = text;

  if (text.length < 3) {
    console.debug("RyView ignored a very short selection:", text);
    return;
  }

  const rect = getSelectionRect();
  if (!rect) return;

  lastCall = now;
  showLoading(rect);

  try {
    const data = await predictReview(text);
    showPopup(data.label, data.confidence, text, data.token_scores || []);
  } catch (err) {
    console.error("RyView API error:", err);
    showError(rect, err);
  }
}

document.addEventListener("mouseup", () => {
  setTimeout(() => {
    processSelection();
  }, 300);
});

document.addEventListener("selectionchange", () => {
  setTimeout(() => {
    processSelection();
  }, 150);
});


function createPopupShell(rect) {
  const old = document.getElementById("rg-popup");
  if (old) old.remove();

  const popupW = 300;
  const box = document.createElement("div");
  box.id = "rg-popup";
  box.style.cssText = `
        position: absolute;
        left: ${window.scrollX + rect.left + rect.width / 2 - popupW / 2}px;
        top: ${window.scrollY + rect.top}px;
        transform: translateY(-110%);
        z-index: 999999;
        width: ${popupW}px;
        background: #13161C;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.45);
        font-family: 'Segoe UI', sans-serif;
        overflow: hidden;
    `;

  document.body.appendChild(box);
  activePopup = box;
  return box;
}

function showLoading(rect) {
  const box = createPopupShell(rect);
  box.innerHTML = `
    <div style="height:2px;background:linear-gradient(90deg,transparent,#9B6DFF,transparent);"></div>
    <div style="padding:16px 18px;color:#E8EAF0;font-size:13px;font-weight:600;">
      Analysing review...
    </div>
  `;
}

function showError(rect, err) {
  const box = createPopupShell(rect);
  box.innerHTML = `
    <div style="height:2px;background:linear-gradient(90deg,transparent,#FF4D4D,transparent);"></div>
    <div style="padding:16px 18px;">
      <div style="color:#FF4D4D;font-size:13px;font-weight:700;margin-bottom:6px;">RyView could not finish</div>
      <div style="color:#6B7080;font-size:12px;line-height:1.5;">
        ${String(err?.message || err || "Prediction failed")}
      </div>
    </div>
  `;
}


function showPopup(label, confidence, text, tokenScores) {
  const selection = window.getSelection();
  if (!selection.rangeCount) return;

  const rect = selection.getRangeAt(0).getBoundingClientRect();
  const popupW = 300;
  const isDecep = label === "Deceptive";
  const color = isDecep ? "#FF4D4D" : "#00C48C";
  const icon = isDecep ? "⚠" : "✓";
  const confPct = (confidence * 100).toFixed(1);

  const box = createPopupShell(rect);

  // Position centered above selection
  box.style.cssText = `
        position: absolute;
        left: ${window.scrollX + rect.left + rect.width / 2 - popupW / 2}px;
        top: ${window.scrollY + rect.top}px;
        transform: translateY(-110%);
        z-index: 999999;
        width: ${popupW}px;
        background: #13161C;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.45);
        font-family: 'Segoe UI', sans-serif;
        overflow: hidden;
    `;

  box.innerHTML = `
        <!-- Top accent bar -->
        <div style="height:2px;background:linear-gradient(90deg,transparent,${color},transparent);"></div>

        <div style="padding:16px 18px;">

            <!-- Brand row -->
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
              <span style="display:inline-flex;align-items:center;gap:8px;font-size:12px;font-weight:700;color:#E8EAF0;letter-spacing:-.2px;">
                <span style="width:18px;height:18px;border-radius:6px;background:linear-gradient(135deg,#9B6DFF,#7C3AED);display:inline-flex;align-items:center;justify-content:center;color:white;font-size:11px;font-weight:800;">R</span>
                RyView
                </span>
                <span style="
                    font-size:11px;font-weight:700;
                    color:${color};
                    background:${isDecep ? 'rgba(255,77,77,0.1)' : 'rgba(0,196,140,0.1)'};
                    border:1px solid ${color};
                    border-radius:999px;
                    padding:2px 10px;
                ">${icon} ${label}</span>
            </div>

            <!-- Confidence bar -->
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;font-size:11px;color:#6B7080;margin-bottom:5px;">
                    <span>Confidence</span>
                    <span style="color:${color};font-weight:700;">${confPct}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:5px;overflow:hidden;">
                    <div style="
                        width:${confPct}%;
                        height:100%;
                        background:${color};
                        border-radius:999px;
                        transition:width .6s ease;
                    "></div>
                </div>
            </div>

            <!-- View Details button -->
            <button id="rg-details-btn" style="
                width:100%; padding:9px;
                border:none; border-radius:9px;
                background:${color};
                color:white; font-weight:600;
                font-size:13px; cursor:pointer;
                font-family:inherit;
            ">
                View Full Explanation →
            </button>

        </div>
    `;

  // Pass text + token scores to dashboard
  document.getElementById("rg-details-btn").addEventListener("click", () => {
    const params = new URLSearchParams({
      label: label,
      confidence: confPct,
      text: text,
      token_scores: JSON.stringify(tokenScores)
    });
    window.open(`http://127.0.0.1:8001/dashboard?${params.toString()}`, "_blank");
  });

  // Auto-dismiss after 12 seconds
  setTimeout(() => {
    if (document.getElementById("rg-popup")) box.remove();
  }, 12000);
}