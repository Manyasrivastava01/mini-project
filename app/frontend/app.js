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

function setOutput(obj) {
  document.getElementById("output").textContent = pretty(obj);
}

function parseJsonFromTextarea() {
  const raw = document.getElementById("inputJson").value.trim();
  if (!raw) throw new Error("Input is empty");
  return JSON.parse(raw);
}

document.getElementById("btnSample").addEventListener("click", async () => {
  setOutput({ loading: true });

  try {
    const s = await apiGet("/sample");
    if (s.error) {
      setOutput(s);
      return;
    }

    // Put only the features dict into the textbox (user-friendly)
    document.getElementById("inputJson").value = pretty(s.features);

    setOutput({
      ok: true,
      message: "Loaded sample row into textbox",
      source_file: s.source_file,
      row_index: s.row_index,
      n_features: s.n_features
    });
  } catch (e) {
    setOutput({ error: String(e) });
  }
});

document.getElementById("btnPredict").addEventListener("click", async () => {
  setOutput({ loading: true });

  try {
    const feats = parseJsonFromTextarea();
    const applyNorm = document.getElementById("applyNorm").checked;

    const result = await apiPost("/predict", {
      features: feats,
      apply_normalization: applyNorm
    });

    // Put the warning up top if present
    if (result.warning) {
      result.note = result.warning;
    }

    setOutput(result);
  } catch (e) {
    setOutput({ error: String(e) });
  }
});
