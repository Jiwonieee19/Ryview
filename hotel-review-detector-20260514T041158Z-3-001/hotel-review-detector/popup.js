document.getElementById("check").addEventListener("click", async () => {

    console.log("Button clicked");

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {

        chrome.tabs.sendMessage(
            tabs[0].id,
            { action: "GET_SELECTED_TEXT" },
            async (response) => {
                const error = chrome.runtime.lastError;
                if (error) {
                    document.getElementById("result").innerText = error.message;
                    return;
                }

                const reviewText = response?.text || "";

                if (!reviewText) {
                    document.getElementById("result").innerText =
                        "Please highlight a review first.";
                    return;
                }

                chrome.runtime.sendMessage(
                    { action: "PREDICT_REVIEW", text: reviewText },
                    (predictionResponse) => {
                        const predictionError = chrome.runtime.lastError;
                        if (predictionError) {
                            document.getElementById("result").innerText = predictionError.message;
                            return;
                        }

                        if (!predictionResponse?.ok) {
                            document.getElementById("result").innerText =
                                predictionResponse?.error || "Prediction failed.";
                            return;
                        }

                        document.getElementById("result").innerText =
                            "Prediction: " + predictionResponse.data.label;
                    }
                );
            }
        );
    });
});