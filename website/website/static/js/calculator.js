// calculator.js -- wires up the two-tab what-if simulator to the Flask
// /api/predict/uci and /api/predict/brfss endpoints, mirroring the live
// slider -> prediction -> SHAP chart behavior of the original Streamlit app.

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function renderResult(boxId, chartId, data) {
  const box = document.getElementById(boxId);
  box.innerHTML = `
    <div class="risk-box" style="background:${data.color}22;border:2px solid ${data.color};">
      <span class="risk-pct" style="color:${data.color}">${data.risk_pct.toFixed(1)}%</span>
      <span class="risk-band" style="color:${data.color}">${data.band}</span>
    </div>
    <p class="result-note">This is a model estimate from training data, not a clinical probability. See the project report for the calibration analysis.</p>
  `;
  const chartBox = document.getElementById(chartId);
  chartBox.innerHTML = `<img class="shap-img" src="${data.chart}" alt="SHAP waterfall explanation">
    <p class="result-note">Each bar shows how much that feature value pushed this prediction up (red) or down (blue) from the model's average.</p>`;
}

function renderError(boxId, chartId, msg) {
  document.getElementById(boxId).innerHTML = `<p class="loading">Error: ${msg}</p>`;
  document.getElementById(chartId).innerHTML = "";
}

async function updateUci() {
  const sex = document.querySelector('input[name="uci_sex"]:checked').value;
  const exang = document.querySelector('input[name="uci_exang"]:checked').value;
  const fbs = document.querySelector('input[name="uci_fbs"]:checked').value;
  const payload = {
    age: document.getElementById("uci_age").value,
    sex, exang, fbs,
    cp: document.getElementById("uci_cp").value,
    restecg: document.getElementById("uci_restecg").value,
    thal: document.getElementById("uci_thal").value,
    slope: document.getElementById("uci_slope").value,
    thalach: document.getElementById("uci_thalach").value,
    oldpeak: document.getElementById("uci_oldpeak").value,
    ca: document.getElementById("uci_ca").value,
    chol: document.getElementById("uci_chol").value,
    trestbps: document.getElementById("uci_trestbps").value,
  };
  try {
    const res = await fetch("/api/predict/uci", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    const data = await res.json();
    renderResult("uci_result_box", "uci_chart_box", data);
  } catch (e) {
    renderError("uci_result_box", "uci_chart_box", e.message);
  }
}

async function updateBrfss() {
  const sex = document.querySelector('input[name="brfss_sex"]:checked').value;
  const highbp = document.querySelector('input[name="brfss_highbp"]:checked').value;
  const highchol = document.querySelector('input[name="brfss_highchol"]:checked').value;
  const diffwalk = document.querySelector('input[name="brfss_diffwalk"]:checked').value;
  const stroke = document.querySelector('input[name="brfss_stroke"]:checked').value;
  const smoker = document.querySelector('input[name="brfss_smoker"]:checked').value;
  const physact = document.querySelector('input[name="brfss_physact"]:checked').value;
  const fruits = document.querySelector('input[name="brfss_fruits"]:checked').value;
  const veggies = document.querySelector('input[name="brfss_veggies"]:checked').value;
  const hvyalcohol = document.querySelector('input[name="brfss_alcohol"]:checked').value;
  const payload = {
    sex, highbp, highchol, diffwalk, stroke, smoker, physact, fruits, veggies, hvyalcohol,
    age_band: document.getElementById("brfss_age").value,
    genhlth: document.getElementById("brfss_genhlth").value,
    education: document.getElementById("brfss_education").value,
    income: document.getElementById("brfss_income").value,
    diabetes: document.getElementById("brfss_diabetes").value,
    bmi: document.getElementById("brfss_bmi").value,
  };
  try {
    const res = await fetch("/api/predict/brfss", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    const data = await res.json();
    renderResult("brfss_result_box", "brfss_chart_box", data);
  } catch (e) {
    renderError("brfss_result_box", "brfss_chart_box", e.message);
  }
}

const debouncedUci = debounce(updateUci, 350);
const debouncedBrfss = debounce(updateBrfss, 350);

function wireTab(prefix, updateFn) {
  const panel = document.getElementById(`tab-${prefix}`);
  panel.querySelectorAll("input, select").forEach((el) => {
    const evt = el.tagName === "SELECT" || el.type === "radio" ? "change" : "input";
    el.addEventListener(evt, updateFn);
  });
  // live-update the numeric labels next to range sliders
  panel.querySelectorAll('input[type="range"]').forEach((slider) => {
    const label = document.getElementById(slider.id + "_val");
    if (label) slider.addEventListener("input", () => (label.textContent = slider.value));
  });
}

wireTab("uci", debouncedUci);
wireTab("brfss", debouncedBrfss);

// Tab switching
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// Initial predictions on page load
updateUci();
updateBrfss();
