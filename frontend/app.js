// GreenGrid AI — Dashboard JavaScript
// Calls POST /analyze on the FastAPI backend and renders the results.

const API_URL = "http://127.0.0.1:8001/analyze";

async function runAnalysis() {
  const btn    = document.getElementById("run-btn");
  const msgEl  = document.getElementById("status-msg");

  // Gather inputs
  const payload = {
    timestamp         : new Date().toISOString().slice(0, 19).replace("T", " "),
    temperature       : parseFloat(document.getElementById("temperature").value),
    building_type     : document.getElementById("building_type").value,
    occupants         : parseInt(document.getElementById("occupants").value, 10),
    // Sliders are now in model units (kW). The backend expects 0-100 percentages
    // and scales them back: ac% / 100 * 20 and appliance% / 100 * 10.
    // So we reverse that: send (kw / max) * 100 to recover the correct kW on the other side.
    ac_usage          : parseFloat(document.getElementById("ac_usage").value)        / 11 * 100,
    appliance_usage   : parseFloat(document.getElementById("appliance_usage").value) / 7  * 100,
    peak_hour         : parseInt(document.getElementById("peak_hour").value, 10),
    actual_consumption: parseFloat(document.getElementById("actual_consumption").value),
  };

  // Basic client-side guard
  if (isNaN(payload.actual_consumption) || payload.actual_consumption <= 0) {
    msgEl.textContent = "Please enter a valid actual consumption value (> 0).";
    return;
  }

  // Show loading state
  btn.disabled    = true;
  btn.textContent = "Analyzing…";
  msgEl.textContent = "Sending request to backend…";
  document.getElementById("results-placeholder").style.display = "none";
  document.getElementById("results-content").style.display     = "none";

  try {
    const resp = await fetch(API_URL, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify(payload),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      const detail = err.detail
        ? (Array.isArray(err.detail)
            ? err.detail.map(e => `${e.loc?.slice(-1)[0]}: ${e.msg}`).join(" | ")
            : err.detail)
        : `HTTP ${resp.status}`;
      throw new Error(detail);
    }

    const data = await resp.json();
    renderResults(data);
    msgEl.textContent = "Analysis complete.";

  } catch (err) {
    document.getElementById("results-placeholder").style.display = "block";
    document.getElementById("results-placeholder").innerHTML =
      `<span style="color:#c0392b">⚠ Error: ${err.message}</span>`;
    msgEl.textContent = "";
  } finally {
    btn.disabled    = false;
    btn.textContent = "Run Analysis";
  }
}

function renderResults(d) {
  // ── Actual vs Predicted bars ────────────────────────────────────────────
  const maxVal = Math.max(d.actual_consumption, d.predicted_consumption, 0.01);
  const actualPct    = Math.round((d.actual_consumption    / maxVal) * 100);
  const predictedPct = Math.round((d.predicted_consumption / maxVal) * 100);

  document.getElementById("bar-actual").style.width    = actualPct    + "%";
  document.getElementById("bar-predicted").style.width = predictedPct + "%";
  document.getElementById("num-actual").textContent    = d.actual_consumption.toFixed(2)    + " kWh";
  document.getElementById("num-predicted").textContent = d.predicted_consumption.toFixed(2) + " kWh";

  // ── Key metrics ─────────────────────────────────────────────────────────
  document.getElementById("m-deviation").textContent =
    (d.deviation_percent >= 0 ? "+" : "") + d.deviation_percent.toFixed(1) + "%";

  const statusEl = document.getElementById("m-status");
  statusEl.textContent = d.status;
  statusEl.className   = "metric-value metric-status " + {
    Normal  : "status-normal",
    Elevated: "status-elevated",
    Abnormal: "status-abnormal",
  }[d.status] || "";

  document.getElementById("m-savings").textContent = d.estimated_savings || "—";

  // ── Possible factors ────────────────────────────────────────────────────
  const factorsList = document.getElementById("factors-list");
  factorsList.innerHTML = "";
  const factors = d.possible_factors || [];
  if (factors.length === 0) {
    factorsList.innerHTML = "<li>None detected</li>";
  } else {
    factors.forEach(f => {
      const li = document.createElement("li");
      li.textContent = f;
      factorsList.appendChild(li);
    });
  }

  // ── RAG guidance ────────────────────────────────────────────────────────
  const ragList = document.getElementById("rag-list");
  ragList.innerHTML = "";
  const guidance = d.rag_guidance || [];
  if (guidance.length === 0) {
    ragList.innerHTML = "<p style='color:#8a9a8a;font-size:.88rem'>No guidance retrieved.</p>";
  } else {
    guidance.forEach(g => {
      const div = document.createElement("div");
      div.className = "rag-item";
      div.innerHTML = `
        <div class="rag-item-header">
          <span class="rag-source">${escHtml(g.source)}</span>
          <span class="rag-score">score: ${g.score.toFixed(3)}</span>
        </div>
        <div class="rag-section">${escHtml(g.section)}</div>
        <div class="rag-text">${escHtml(g.text.replace(/^SECTION:[^\n]*\n/, ""))}</div>
      `;
      ragList.appendChild(div);
    });
  }

  // ── AI recommendation ────────────────────────────────────────────────────
  const recBox = document.getElementById("recommendation-box");
  if (d.recommendation) {
    recBox.className   = "recommendation-box";
    recBox.textContent = d.recommendation;
  } else {
    recBox.className   = "recommendation-box recommendation-na";
    recBox.textContent = "Recommendation not available — Ollama may not be running.\n"
                       + "Start Ollama with:  ollama serve";
  }

  // Show results panel
  document.getElementById("results-content").style.display = "block";
}

// Escape user-facing text to prevent XSS from API strings
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
