const resultEl = document.getElementById("result");
const checkBtn = document.getElementById("check");

async function getSelectedText(tabId) {
    try {
        const response = await chrome.tabs.sendMessage(tabId, { action: "GET_SELECTED_TEXT" });
        if (response?.text) {
            return response.text.trim();
        }
    } catch {
        // fall back below
    }

    const [result] = await chrome.scripting.executeScript({
        target: { tabId },
        func: () => window.getSelection().toString().trim()
    });

    return result?.result || "";
}

checkBtn.addEventListener("click", async () => {
    resultEl.textContent = "";

    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
        const tab = tabs?.[0];
        if (!tab?.id) {
            resultEl.textContent = "No active tab found.";
            return;
        }

        try {
            const reviewText = await getSelectedText(tab.id);

            if (!reviewText) {
                resultEl.textContent = "Please highlight a review first.";
                return;
            }

            chrome.runtime.sendMessage(
                { action: "PREDICT_REVIEW", text: reviewText },
                (predictionResponse) => {
                    const predictionError = chrome.runtime.lastError;
                    if (predictionError) {
                        resultEl.textContent = predictionError.message;
                        return;
                    }

                    if (!predictionResponse?.ok) {
                        resultEl.textContent = predictionResponse?.error || "Prediction failed.";
                        return;
                    }

                    resultEl.textContent = "Prediction: " + predictionResponse.data.label;
                }
            );
        } catch (error) {
            resultEl.textContent = error?.message || String(error);
        }
    });
});
