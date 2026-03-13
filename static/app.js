const state = {
  mode: "simple",
  map: null,
  layerControl: null,
  overlays: {},
  baseLayers: {},
};

const colors = {
  vegetation: "#43a557",
  water: "#2b7edb",
  urban: "#e88c3a",
  ice: "#a4d5f2",
};

function setMode(mode) {
  state.mode = mode;
  document.getElementById("modeSimple").classList.toggle("active", mode === "simple");
  document.getElementById("modeExpert").classList.toggle("active", mode === "expert");
}

function initMap() {
  state.map = L.map("map", { zoomControl: false }).setView([37.7749, -122.4194], 9);
  L.control.zoom({ position: "bottomright" }).addTo(state.map);

  const streets = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap",
  });
  const satellite = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { attribution: "Tiles © Esri" }
  );

  streets.addTo(state.map);
  state.baseLayers = {
    "Streets": streets,
    "Satellite": satellite,
  };

  state.layerControl = L.control.layers(state.baseLayers, {}, { collapsed: false }).addTo(state.map);
}

function updateOverlays(overlays, bbox) {
  const bounds = L.latLngBounds([bbox.south, bbox.west], [bbox.north, bbox.east]);

  Object.values(state.overlays).forEach(layer => {
    state.map.removeLayer(layer);
  });
  state.overlays = {};

  if (state.layerControl) {
    state.layerControl.remove();
  }
  state.layerControl = L.control.layers(state.baseLayers, {}, { collapsed: false }).addTo(state.map);

  Object.keys(overlays).forEach(key => {
    const overlay = L.imageOverlay(overlays[key], bounds, { opacity: 0.65 });
    overlay.addTo(state.map);
    state.layerControl.addOverlay(overlay, `${key.charAt(0).toUpperCase() + key.slice(1)} change`);
    state.overlays[key] = overlay;
  });

  state.map.fitBounds(bounds, { padding: [30, 30] });
}

function updateStats(changes) {
  document.querySelector("#statVeg .value").textContent = `${changes.vegetation.toFixed(1)} pp`;
  document.querySelector("#statWater .value").textContent = `${changes.water.toFixed(1)} pp`;
  document.querySelector("#statUrban .value").textContent = `${changes.urban.toFixed(1)} pp`;
  document.querySelector("#statIce .value").textContent = `${changes.ice.toFixed(1)} pp`;
}

function updateSummary(summary, recommendations) {
  document.getElementById("summaryText").textContent = summary;
  const container = document.getElementById("recommendations");
  container.innerHTML = "";
  recommendations.forEach(rec => {
    const chip = document.createElement("span");
    chip.textContent = rec;
    container.appendChild(chip);
  });
}

function plotSeries(container, title, years, series, forecastYears, forecast, color) {
  const trace = {
    x: years,
    y: series,
    mode: "lines+markers",
    line: { color },
    name: "Observed",
  };
  const forecastTrace = {
    x: forecastYears,
    y: forecast,
    mode: "lines",
    line: { color, dash: "dot" },
    name: "Forecast",
  };

  const layout = {
    margin: { l: 40, r: 20, t: 10, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    showlegend: false,
    xaxis: { title: "Year" },
    yaxis: { title: title },
    font: { family: "Space Grotesk" },
  };

  Plotly.newPlot(container, [trace, forecastTrace], layout, { displayModeBar: false });
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error("Request failed");
  }
  return res.json();
}

async function analyze() {
  const button = document.getElementById("analyzeBtn");
  button.disabled = true;
  button.textContent = "Analyzing...";

  try {
    const query = document.getElementById("locationInput").value.trim();
    const payload = { query: query || null, mode: state.mode, year_start: 2018, year_end: 2023 };

    const analysis = await postJSON("/analyze_location", payload);
    updateStats(analysis.changes);
    updateSummary(analysis.summary, analysis.recommendations);
    updateOverlays(analysis.overlays, analysis.bbox);

    document.getElementById("locationLabel").textContent = analysis.location.name;
    document.getElementById("yearLabel").textContent = `${analysis.years[0]} vs ${analysis.years[analysis.years.length - 1]}`;
    document.getElementById("previewStart").src = analysis.previews.start;
    document.getElementById("previewEnd").src = analysis.previews.end;
    document.getElementById("previewLabelStart").textContent = `${analysis.years[0]}`;
    document.getElementById("previewLabelEnd").textContent = `${analysis.years[analysis.years.length - 1]}`;

    const charts = await postJSON("/get_charts_data", payload);
    plotSeries("chart-veg", "Vegetation %", charts.years, charts.series.vegetation, charts.forecast_years, charts.forecast.vegetation, colors.vegetation);
    plotSeries("chart-water", "Water %", charts.years, charts.series.water, charts.forecast_years, charts.forecast.water, colors.water);
    plotSeries("chart-urban", "Urban %", charts.years, charts.series.urban, charts.forecast_years, charts.forecast.urban, colors.urban);
    plotSeries("chart-temp", "Temp anomaly", charts.years, charts.series.temperature, charts.forecast_years, charts.forecast.temperature, "#f06d4f");
  } catch (err) {
    document.getElementById("summaryText").textContent = "Analysis failed. Check the backend logs.";
  } finally {
    button.disabled = false;
    button.textContent = "Analyze";
  }
}

async function askQuestion() {
  const question = document.getElementById("questionInput").value.trim();
  if (!question) {
    return;
  }
  const payload = { query: document.getElementById("locationInput").value.trim() || null, mode: state.mode, question };
  try {
    const response = await postJSON("/ai_explanation", payload);
    document.getElementById("answerText").textContent = response.summary;
  } catch (err) {
    document.getElementById("answerText").textContent = "AI assistant is unavailable.";
  }
}

window.addEventListener("load", () => {
  initMap();
  document.getElementById("modeSimple").addEventListener("click", () => setMode("simple"));
  document.getElementById("modeExpert").addEventListener("click", () => setMode("expert"));
  document.getElementById("analyzeBtn").addEventListener("click", analyze);
  document.getElementById("askBtn").addEventListener("click", askQuestion);
  analyze();
});
