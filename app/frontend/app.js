
async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return await res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`POST ${path} failed: ${res.status}\n${txt}`);
  }
  return await res.json();
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function setRawOutput(obj) {
  document.getElementById("output").textContent = pretty(obj);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function parseJsonFromTextarea() {
  const raw = document.getElementById("inputJson").value.trim();
  if (!raw) throw new Error("Input is empty");
  return JSON.parse(raw);
}

function bandClass(band) {
  const b = String(band || "").toLowerCase();
  if (b === "high") return "pill pill-high";
  if (b === "medium") return "pill pill-medium";
  return "pill pill-low";
}

function renderHumanOutput(result) {
  const host = document.getElementById("humanOutput");

  if (!result || result.error) {
    host.innerHTML = `<div class="error">${escapeHtml(result?.error || "Unknown error")}</div>`;
    return;
  }

  const score = (typeof result.score === "number") ? result.score : null;
  const scoreDisplay = (score === null) ? "—" : String(Math.max(0, score).toFixed(1));
  const scoreRawNote = (score !== null && score < 0) ? `Raw model score: ${score.toFixed(1)} (clamped to 0 for display)` : "";
  const band = result.band || "";
  const t = result.thresholds || {};
  const mt = t.medium_threshold ?? null;
  const ht = t.high_threshold ?? null;
  const mode = t.mode ? String(t.mode) : "";

  const causalState = Array.isArray(result.causal_state) ? result.causal_state : [];
  const reasons = Array.isArray(result.top_reasons) ? result.top_reasons : [];
  const protective = Array.isArray(result.protective_signals) ? result.protective_signals : [];
  const mitigations = Array.isArray(result.mitigation_suggestions) ? result.mitigation_suggestions : [];

  const scoreHtml = scoreDisplay;

  const thresholdsLine =
    (mt === null || ht === null)
      ? `<span class="muted">Thresholds not available</span>`
      : `Low &lt; ${escapeHtml(mt)} | Medium ≥ ${escapeHtml(mt)} and &lt; ${escapeHtml(ht)} | High ≥ ${escapeHtml(ht)}${mode ? ` <span class="muted">(mode: ${escapeHtml(mode)})</span>` : ""}`;

  const patternsHtml = (causalState.length === 0)
    ? `<div class="muted">No strong fatigue-increasing causal patterns detected for this window.</div>`
    : `<ul class="bullets">
        ${causalState.map(p => `
          <li>
            <b>${escapeHtml(p.title || "")}</b>
            <div class="muted small">${escapeHtml(p.details || "")}</div>
            ${p.evidence ? `<div class="evidence">Evidence: ${escapeHtml(p.evidence)}</div>` : ""}
          </li>
        `).join("")}
      </ul>`;


const driversHtml = (reasons.length === 0)
    ? `<div class="muted">No fatigue-increasing driver features detected (given activation threshold).</div>`
    : `<ul class="bullets">
        ${reasons.map(r => `
          <li>
            <div class="driver-line">
              <b>${escapeHtml(r.feature_label || r.feature || "")}</b>
              <span class="driver-tags">
                ${r.direction ? `<span class="tag tag-dir">${escapeHtml(String(r.direction).toUpperCase())}</span>` : ``}
                ${r.confidence ? `<span class="tag tag-conf">${escapeHtml(r.confidence)}</span>` : ``}
                ${r.strength ? `<span class="tag tag-str">${escapeHtml(r.strength)}</span>` : ``}
                ${(r.subject_frac !== undefined && r.subject_frac !== null) ? `<span class="tag tag-subj">${escapeHtml((Number(r.subject_frac)*100).toFixed(0))}% subj</span>` : ``}
              </span>
            </div>
            <div class="muted small">${escapeHtml(r.message || "")}</div>
          </li>
        `).join("")}
      </ul>`;


const protectiveHtml = (protective.length === 0)
    ? `<div class="muted">No protective / counteracting signals detected.</div>`
    : `<ul class="bullets">
        ${protective.map(r => `
          <li>
            <div class="driver-line">
              <b>${escapeHtml(r.feature_label || r.feature || "")}</b>
              <span class="driver-tags">
                ${r.direction ? `<span class="tag tag-dir">${escapeHtml(String(r.direction).toUpperCase())}</span>` : ``}
                ${r.confidence ? `<span class="tag tag-conf">${escapeHtml(r.confidence)}</span>` : ``}
                ${r.strength ? `<span class="tag tag-str">${escapeHtml(r.strength)}</span>` : ``}
                ${(r.subject_frac !== undefined && r.subject_frac !== null) ? `<span class="tag tag-subj">${escapeHtml((Number(r.subject_frac)*100).toFixed(0))}% subj</span>` : ``}
              </span>
            </div>
            <div class="muted small">${escapeHtml(r.message || "")}</div>
          </li>
        `).join("")}
      </ul>`;

  const mitigationHtml = (mitigations.length === 0)
    ? `<div class="muted">No mitigation suggestions available.</div>`
    : `<ul class="bullets">
        ${mitigations.map(m => `<li>${escapeHtml(m)}</li>`).join("")}
      </ul>`;

  const inputCheck = result.note || result.warning || "";

  host.innerHTML = `
    <div class="score-card">
      <div class="score">${escapeHtml(scoreHtml)}</div>
      <div class="score-meta">
        <div class="muted">Fatigue score (regression output)</div>
        ${scoreRawNote ? `<div class="muted small">${escapeHtml(scoreRawNote)}</div>` : ``}
        <div class="${bandClass(band)}">${escapeHtml(band)} RISK</div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Risk thresholds</div>
      <div class="small">${thresholdsLine}</div>
    </div>

    <div class="section">
      <div class="section-title">What is happening (causal state)</div>
      ${patternsHtml}
    </div>

    <div class="section">
      <div class="section-title">Why this matters (causal reasoning)</div>\n      <div class="small"><a href="/causal_graph.png" target="_blank" rel="noopener">View causal graph (PNG)</a> · <a href="/causal_graph.json" target="_blank" rel="noopener">JSON</a></div>
      <div class="small">${escapeHtml(result.causal_reasoning || "—")}</div>
    </div>

    <div class="section">
      <div class="section-title">What can help (mitigation suggestions)</div>
      ${mitigationHtml}
      <div class="muted small disclaimer">These are general, non-medical suggestions. Use domain and safety judgment.</div>
    </div>

    <div class="section">
      <div class="section-title">Feature-level evidence (fatigue-increasing drivers)</div>
      ${driversHtml}
    </div>

    <div class="section">
      <div class="section-title">Protective / counteracting signals</div>
      ${protectiveHtml}
    </div>

    <div class="section">
      <div class="section-title">Input check</div>
      <div class="small">${escapeHtml(inputCheck)}</div>
    </div>
  `;
}

document.getElementById("btnSample").addEventListener("click", async () => {
  setRawOutput({ loading: true });
  renderHumanOutput({});

  try {
    const s = await apiGet("/sample");
    if (s.error) {
      setRawOutput(s);
      renderHumanOutput(s);
      return;
    }

    // Put only the features dict into the textbox (user-friendly)
    document.getElementById("inputJson").value = pretty(s.features);

    const info = {
      ok: true,
      message: "Loaded sample row into textbox",
      source_file: s.source_file,
      row_index: s.row_index,
      n_features: s.n_features,
    };
    setRawOutput(info);
    renderHumanOutput({ score: null, band: "", thresholds: {}, note: info.message });
  } catch (e) {
    const err = { error: String(e) };
    setRawOutput(err);
    renderHumanOutput(err);
  }
});

document.getElementById("btnPredict").addEventListener("click", async () => {
  setRawOutput({ loading: true });
  renderHumanOutput({ score: null, band: "", thresholds: {}, note: "Running prediction..." });

  try {
    const feats = parseJsonFromTextarea();
    const applyNorm = document.getElementById("applyNorm").checked;

    const result = await apiPost("/predict", {
      features: feats,
      apply_normalization: applyNorm,
    });

    setRawOutput(result);
    renderHumanOutput(result);
  } catch (e) {
    const err = { error: String(e) };
    setRawOutput(err);
    renderHumanOutput(err);
  }
});
