const API_BASE = "http://127.0.0.1:8001";

async function checkAPI() {
  const dot = document.getElementById("api-dot");
  const txt = document.getElementById("api-txt");
  try {
    const res = await fetch(`${API_BASE}/`, {
      signal: AbortSignal.timeout(2000)
    });
    if (res.ok) {
      dot.className = "dot dot-on";
      txt.textContent = "API Online";
    } else throw new Error();
  } catch {
    dot.className = "dot dot-off";
    txt.textContent = "API Offline";
  }
}

checkAPI();
