console.log("ReviewGuard content script loaded.");

let lastCall = 0;

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

document.addEventListener("mouseup", () => {

    setTimeout(async () => {

        const now = Date.now();
        if (now - lastCall < 2000) return;

        const text = window.getSelection().toString().trim();
        if (!text || text.length < 20) return;

        lastCall = now;

        try {
            const data = await predictReview(text);
            showPopup(data.label, data.confidence, text, data.token_scores || []);
        } catch (err) {
            console.error("ReviewGuard API error:", err);
            return;
        }

    }, 300);
});


function showPopup(label, confidence, text, tokenScores) {

    // Remove existing popup
    const old = document.getElementById("rg-popup");
    if (old) old.remove();

    const selection = window.getSelection();
    if (!selection.rangeCount) return;

    const rect = selection.getRangeAt(0).getBoundingClientRect();
    const popupW = 300;
    const isDecep = label === "Deceptive";
    const color = isDecep ? "#FF4D4D" : "#00C48C";
    const icon = isDecep ? "⚠" : "✓";
    const confPct = (confidence * 100).toFixed(1);

    // Build mini highlights for popup (top 5 words only)
    const topWords = [...tokenScores]
        .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
        .slice(0, 5)
        .filter(w => w.word.trim());

    const topWordsHTML = topWords.map(w => {
        const wColor = w.score > 0
            ? (isDecep ? "#FF4D4D" : "#00C48C")
            : (isDecep ? "#00C48C" : "#FF4D4D");
        const arrow = w.score > 0 ? "↑" : "↓";
        return `<span style="
            display:inline-flex; align-items:center; gap:4px;
            background:rgba(255,255,255,0.06);
            border-radius:999px; padding:3px 10px;
            font-size:11px; color:#E8EAF0;
            margin:3px 3px 0 0;
        ">
            ${w.word}
            <span style="color:${wColor};font-weight:700;">${arrow}${Math.abs(w.score).toFixed(2)}</span>
        </span>`;
    }).join("");

    const box = document.createElement("div");
    box.id = "rg-popup";

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
                <span style="font-size:12px;font-weight:700;color:#E8EAF0;letter-spacing:-.2px;">
                    🛡 ReviewGuard
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

            <!-- Top words -->
            ${topWordsHTML ? `
            <div style="margin-bottom:14px;">
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#6B7080;margin-bottom:6px;">
                    Key influencing words
                </div>
                <div>${topWordsHTML}</div>
            </div>` : ''}

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

    document.body.appendChild(box);

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