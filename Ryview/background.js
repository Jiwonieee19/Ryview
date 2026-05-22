const API_BASE = "http://127.0.0.1:8001";

async function proxyPredict(text) {
    const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`API ${res.status}: ${body}`);
    }
    return res.json();
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    // Support both our earlier `{ type: 'predict', payload: { text } }` shape
    // and the current content script `{ action: 'PREDICT_REVIEW', text }
    try {
        if (!msg) {
            sendResponse({ ok: false, error: 'Empty message' });
            return;
        }

        if (msg.type === 'predict' && msg.payload && msg.payload.text) {
            proxyPredict(msg.payload.text)
                .then(data => sendResponse({ ok: true, data }))
                .catch(err => sendResponse({ ok: false, error: String(err) }));
            return true;
        }

        if (msg.action === 'PREDICT_REVIEW' && msg.text) {
            proxyPredict(msg.text)
                .then(data => sendResponse({ ok: true, data }))
                .catch(err => sendResponse({ ok: false, error: String(err) }));
            return true;
        }

        sendResponse({ ok: false, error: 'Unknown message format' });
    } catch (e) {
        sendResponse({ ok: false, error: String(e) });
    }
});
