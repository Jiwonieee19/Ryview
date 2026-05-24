async function predictReview(text) {
    const res = await fetch("http://127.0.0.1:8001/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });

    if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(`API request failed (${res.status}): ${body || res.statusText}`);
    }

    return res.json();
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message?.action !== "PREDICT_REVIEW") {
        return;
    }

    predictReview(message.text)
        .then((data) => sendResponse({ ok: true, data }))
        .catch((error) => {
            console.error("ReviewGuard API error:", error);
            sendResponse({ ok: false, error: error.message || String(error) });
        });

    return true;
});
