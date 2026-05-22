/**
 * Ryview — content.js
 * User highlights any review text → popup appears with
 * verdict, confidence bar, word saliency, and View Details button.
 */

const API_BASE = "http://127.0.0.1:8001";

// ── Color palette (dark purple) ──────────────────────────────────────────────
const PALETTE = {
  bg: "#110D1A",
  surface: "#1C1628",
  surface2: "#241E33",
  border: "rgba(255,255,255,0.08)",
  text: "#EAE6F0",
  muted: "#7A7290",
  accent: "#9B6DFF",
  accentSoft: "rgba(155,109,255,0.12)",
  deceptive: "#F0527A",
  deceptiveBg: "rgba(240,82,122,0.10)",
  genuine: "#52C9A0",
  genuineBg: "rgba(82,201,160,0.10)",
};

let lastCall = 0;
let activePopup = null;

// ── Listen for text selection ─────────────────────────────────────────────────
document.addEventListener("mouseup", () => {
  setTimeout(async () => {

    const now = Date.now();
    if (now - lastCall < 1500) return;

    const text = window.getSelection()?.toString().trim();
    if (!text || text.length < 20) return;

    lastCall = now;
    const selection = window.getSelection();
    if (!selection.rangeCount) return;

    const rect = selection.getRangeAt(0).getBoundingClientRect();
    showLoading(rect);

    try {
      // Route the API call through the extension background worker to avoid
      // mixed-content and extension-blocking issues on HTTPS pages.
      const data = await new Promise((resolve, reject) => {
        try {
          chrome.runtime.sendMessage(
            { type: 'predict', payload: { text } },
            (response) => {
              if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError);
              } else {
                resolve(response);
              }
            }
          );
        } catch (e) {
          reject(e);
        }
      });
      if (!data) throw new Error('No response');
      if (data && data.error) throw new Error(data.error);
      showResult(data, text, rect);
    } catch (err) {
      console.error('ReviewGuard API error:', err);
      showError(rect);
    }

  }, 300);
});

// ── Dismiss popup on outside click ───────────────────────────────────────────
document.addEventListener("mousedown", e => {
  if (activePopup && !activePopup.contains(e.target)) {
    activePopup.remove();
    activePopup = null;
  }
});

// ── Position helper ───────────────────────────────────────────────────────────
function positionPopup(el, rect) {
  const W = 310;
  const scrollX = window.scrollX;
  const scrollY = window.scrollY;
  const vpW = window.innerWidth;

  let left = scrollX + rect.left + rect.width / 2 - W / 2;
  let top = scrollY + rect.top;

  if (left < scrollX + 8) left = scrollX + 8;
  if (left + W > scrollX + vpW - 8) left = scrollX + vpW - W - 8;

  el.style.left = `${left}px`;
  el.style.top = `${top}px`;
  el.style.transform = "translateY(-108%)";
}

// ── Create base popup shell ───────────────────────────────────────────────────
function createShell() {
  if (activePopup) activePopup.remove();

  const box = document.createElement("div");
  box.id = "ryview-popup";
  box.style.cssText = `
    position: absolute;
    z-index: 2147483647;
    width: 310px;
    background: ${PALETTE.bg};
    border: 1px solid ${PALETTE.border};
    border-radius: 18px;
    box-shadow: 0 24px 56px rgba(0,0,0,0.55), 0 0 0 1px rgba(155,109,255,0.12);
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden;
    animation: rvFadeUp .22s ease both;
  `;

  if (!document.getElementById("rv-keyframes")) {
    const style = document.createElement("style");
    style.id = "rv-keyframes";
    style.textContent = `
      @keyframes rvFadeUp {
        from { opacity:0; transform:translateY(calc(-108% + 10px)); }
        to   { opacity:1; transform:translateY(-108%); }
      }
      @keyframes rvSpin {
        to { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(style);
  }

  document.body.appendChild(box);
  activePopup = box;
  return box;
}

// ── Loading state ─────────────────────────────────────────────────────────────
function showLoading(rect) {
  const box = createShell();
  box.innerHTML = `
    <div style="
      height: 2px;
      background: linear-gradient(90deg, ${PALETTE.accent}, #C084FC);
      animation: rvScan 1.2s ease-in-out infinite alternate;
    "></div>
    <div style="padding:18px 20px;display:flex;align-items:center;gap:12px;">
      <div style="
        width:18px;height:18px;border-radius:50%;flex-shrink:0;
        border:2px solid ${PALETTE.accent};
        border-top-color:transparent;
        animation:rvSpin .7s linear infinite;
      "></div>
      <div>
        <div style="font-size:13px;font-weight:600;color:${PALETTE.text};">Analysing review…</div>
        <div style="font-size:11px;color:${PALETTE.muted};margin-top:2px;">RyView · RoBERTa</div>
      </div>
    </div>
  `;
  positionPopup(box, rect);
}

// ── Error state ───────────────────────────────────────────────────────────────
function showError(rect) {
  const box = createShell();
  box.innerHTML = `
    <div style="padding:18px 20px;">
      <div style="font-size:13px;font-weight:600;color:#F0527A;">API not reachable</div>
      <div style="font-size:12px;color:${PALETTE.muted};margin-top:4px;">
        Make sure the server is running at<br/>
        <code style="color:${PALETTE.accent};font-size:11px;">http://127.0.0.1:8001</code>
      </div>
    </div>
  `;
  positionPopup(box, rect);
}

// ── Word saliency highlight builder ──────────────────────────────────────────
function buildHighlightedText(tokenScores, isDeceptive) {
  if (!tokenScores || tokenScores.length === 0) return "";

  return tokenScores.map(w => {
    const word = w.word.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const score = w.score;
    const abs = Math.abs(score);
    if (abs < 0.15) return word + " ";

    const positive = score > 0;
    let cls = "";
    if (positive) {
      cls = abs > 0.5
        ? (isDeceptive ? "rv-word-high-decep" : "rv-word-high-genuine")
        : (isDeceptive ? "rv-word-med-decep" : "rv-word-med-genuine");
    } else {
      cls = abs > 0.5
        ? (isDeceptive ? "rv-word-high-genuine" : "rv-word-high-decep")
        : (isDeceptive ? "rv-word-med-genuine" : "rv-word-med-decep");
    }
    return `<span class="${cls}" title="${score.toFixed(3)}">${word}</span> `;
  }).join("");
}

// ── Top influential words ──────────────────────────────────────────────────
function buildTopWords(tokenScores, isDeceptive) {
  if (!tokenScores || tokenScores.length === 0) return "";

  const top = [...tokenScores]
    .filter(w => w.word.trim())
    .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
    .slice(0, 6);

  return top.map(w => {
    const isPos = w.score > 0;
    const color = isPos
      ? (isDeceptive ? PALETTE.deceptive : PALETTE.genuine)
      : (isDeceptive ? PALETTE.genuine : PALETTE.deceptive);
    const arrow = isPos ? "↑" : "↓";
    const direction = isPos ? "Toward" : "Against";
    return `<span style="
      display:inline-flex;align-items:center;gap:4px;
      background:${PALETTE.surface2};
      border:1px solid ${PALETTE.border};
      border-radius:999px;
      padding:3px 10px;
      font-size:11px;color:${PALETTE.text};
      margin:3px 3px 0 0;
    ">${w.word}<span style="color:${color};font-weight:700;" title="${direction} ${isDeceptive ? 'Deceptive' : 'Genuine'}">${direction} · ${arrow}${Math.abs(w.score).toFixed(2)}</span></span>`;
  }).join("");
}

// ── Full result popup ─────────────────────────────────────────────────────────
function showResult(data, text, rect) {
  const { label, confidence, token_scores } = data;
  const isDeceptive = label === "Deceptive";
  const confPct = (confidence * 100).toFixed(1);
  const verdictColor = isDeceptive ? PALETTE.deceptive : PALETTE.genuine;
  const verdictBg = isDeceptive ? PALETTE.deceptiveBg : PALETTE.genuineBg;
  const icon = isDeceptive ? "⚠" : "✓";
  const confOffset = 188.5 * (1 - confidence);

  const highlightedHTML = buildHighlightedText(token_scores, isDeceptive);
  const topWordsHTML = buildTopWords(token_scores, isDeceptive);

  const box = createShell();
  box.innerHTML = `

    <!-- Top gradient bar -->
    <div style="height:2px;background:linear-gradient(90deg,transparent,${verdictColor},transparent);"></div>

    <!-- Header row -->
    <div style="
      padding:14px 18px 12px;
      display:flex;align-items:center;
      justify-content:space-between;
      border-bottom:1px solid ${PALETTE.border};
    ">
      <div style="display:flex;align-items:center;gap:8px;">
        <div style="
          width:28px;height:28px;border-radius:8px;
          background:linear-gradient(135deg,${PALETTE.accent},#7C3AED);
          display:flex;align-items:center;justify-content:center;
          font-size:14px;
        ">𝑅</div>
        <span style="
          font-size:14px;font-weight:700;
          color:${PALETTE.text};letter-spacing:-.2px;
        ">RyView</span>
      </div>
      <div style="
        display:flex;align-items:center;gap:6px;
        background:${verdictBg};
        border:1px solid ${verdictColor};
        border-radius:999px;
        padding:4px 12px;
        font-size:11px;font-weight:700;
        color:${verdictColor};
      ">${icon}&nbsp;${label}</div>
    </div>

    <!-- Verdict row -->
    <div style="padding:16px 18px 14px;display:flex;align-items:center;gap:14px;border-bottom:1px solid ${PALETTE.border};">

      <!-- Confidence ring -->
      <div style="position:relative;width:72px;height:72px;flex-shrink:0;">
        <svg width="72" height="72" viewBox="0 0 72 72" style="transform:rotate(-90deg);">
          <circle cx="36" cy="36" r="30"
            fill="none" stroke="${PALETTE.surface2}" stroke-width="6"/>
          <circle cx="36" cy="36" r="30"
            fill="none" stroke="${verdictColor}" stroke-width="6"
            stroke-linecap="round"
            stroke-dasharray="188.5"
            stroke-dashoffset="${confOffset}"/>
        </svg>
        <div style="
          position:absolute;inset:0;
          display:flex;flex-direction:column;
          align-items:center;justify-content:center;
        ">
          <span style="font-size:15px;font-weight:800;color:${PALETTE.text};line-height:1;">${confPct}%</span>
          <span style="font-size:8px;color:${PALETTE.muted};text-transform:uppercase;letter-spacing:.06em;margin-top:1px;">conf</span>
        </div>
      </div>

      <!-- Verdict text -->
      <div>
        <div style="font-size:20px;font-weight:800;color:${verdictColor};letter-spacing:-.5px;line-height:1;">${label}</div>
        <div style="font-size:12px;color:${PALETTE.muted};margin-top:5px;line-height:1.5;">
          ${isDeceptive
      ? "The model sees language patterns that lean toward a deceptive review."
      : "The model sees language patterns that lean toward a genuine review."}
        </div>
      </div>
    </div>

    <!-- Highlighted text section -->
    ${highlightedHTML ? `
    <div style="padding:14px 18px;border-bottom:1px solid ${PALETTE.border};">
      <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.09em;color:${PALETTE.muted};margin-bottom:8px;">
        Word influence
      </div>
      <div style="font-size:11px;color:${PALETTE.muted};margin-bottom:8px;line-height:1.45;">
        Green means the token helped this verdict. Red means it pushed against it.
      </div>
      <div style="font-size:12.5px;line-height:1.8;color:${PALETTE.text};max-height:90px;overflow-y:auto;">
        ${highlightedHTML}
      </div>
      <div style="display:flex;gap:12px;margin-top:8px;font-size:10px;color:${PALETTE.muted};">
        <span>
          <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${verdictColor};opacity:.7;margin-right:4px;"></span>
          Toward ${label}
        </span>
        <span>
          <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${isDeceptive ? PALETTE.genuine : PALETTE.deceptive};opacity:.6;margin-right:4px;"></span>
          Against ${label}
        </span>
      </div>
    </div>` : ""}

    <!-- Top words -->
    ${topWordsHTML ? `
    <div style="padding:12px 18px;border-bottom:1px solid ${PALETTE.border};">
      <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.09em;color:${PALETTE.muted};margin-bottom:7px;">
        Key words
      </div>
      <div>${topWordsHTML}</div>
    </div>` : ""}

    <!-- Footer -->
    <div style="padding:12px 18px;display:flex;align-items:center;justify-content:space-between;">
      <span style="font-size:11px;color:${PALETTE.muted};">Enhanced RoBERTa · 89.58% accuracy</span>
      <button id="rv-details-btn" style="
        padding:7px 14px;
        background:${PALETTE.accent};
        color:white;font-weight:700;font-size:12px;
        border:none;border-radius:8px;cursor:pointer;
        font-family:inherit;
        transition:opacity .15s;
      ">Full report →</button>
    </div>
  `;

  positionPopup(box, rect);

  // Full dashboard — passes confidence as a plain percentage number
  box.querySelector("#rv-details-btn").addEventListener("click", () => {
    const params = new URLSearchParams({
      label,
      confidence: confPct,          // e.g. "87.4"  (no % sign, already ×100)
      text,
      token_scores: JSON.stringify(token_scores || [])
    });
    window.open(`${API_BASE}/dashboard?${params.toString()}`, "_blank");
  });

  // Hover effect on button
  const btn = box.querySelector("#rv-details-btn");
  btn.addEventListener("mouseenter", () => btn.style.opacity = ".82");
  btn.addEventListener("mouseleave", () => btn.style.opacity = "1");
}