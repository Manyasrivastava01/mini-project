async function doPredict() {
  const out = document.getElementById("out");
  out.textContent = "Loading...";

  let featuresObj;
  try {
    featuresObj = JSON.parse(document.getElementById("jsonBox").value);
  } catch (e) {
    out.textContent = "Invalid JSON in textbox.\n\n" + String(e);
    return;
  }

  const payload = {
    features: featuresObj,
    apply_normalization: document.getElementById("applyNorm").checked
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = "Request failed.\n\n" + String(e);
  }
}

document.getElementById("btnPredict").addEventListener("click", doPredict);
