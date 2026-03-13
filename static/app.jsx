const { useEffect, useRef, useState } = React;

const COLORS = {
  vegetation: "#2f9f6c",
  water: "#2b7edb",
  urban: "#e28b3f",
  ice: "#9bd5f0",
};

const DEFAULT_YEAR_START = 2016;
const DEFAULT_YEAR_END = 2024;
const PROMPTS = [
  "Why is vegetation decreasing here?",
  "What is the biggest change in this region?",
];

const DEBUG = true;

function logDebug(message, payload) {
  if (!DEBUG || !console || !console.debug) {
    return;
  }
  if (payload) {
    console.debug(`[ClimateLens] ${message}`, payload);
  } else {
    console.debug(`[ClimateLens] ${message}`);
  }
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const message = data && data.error && data.error.message ? data.error.message : "Request failed";
    throw new Error(message);
  }
  return data;
}

function computeFactor(year, start, end) {
  const span = Math.max(end - start, 1);
  if (year <= end) {
    return (year - start) / span;
  }
  return 1 + (year - end) / span;
}

function formatChange(value) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(1)} pp`;
}

function safeRenderChart({
  containerId,
  label,
  years,
  series,
  forecastYears,
  forecast,
  color,
  markerYear,
  mode,
}) {
  const el = document.getElementById(containerId);
  if (!el) {
    console.warn(`[ClimateLens] Chart container missing: ${containerId}`, { mode });
    return false;
  }
  if (!Array.isArray(years) || !Array.isArray(series) || years.length === 0 || series.length === 0) {
    logDebug(`Chart skipped (no data): ${containerId}`, {
      hasYears: Array.isArray(years),
      hasSeries: Array.isArray(series),
    });
    return false;
  }

  const trace = {
    x: years,
    y: series,
    mode: "lines+markers",
    line: { color },
    name: "Observed",
  };

  const safeForecastYears = Array.isArray(forecastYears) ? forecastYears : [];
  const safeForecast = Array.isArray(forecast) ? forecast : [];
  const forecastTrace = {
    x: safeForecastYears,
    y: safeForecast,
    mode: "lines",
    line: { color, dash: "dot" },
    name: "Forecast",
  };

  const shapes = markerYear
    ? [
        {
          type: "line",
          x0: markerYear,
          x1: markerYear,
          y0: 0,
          y1: 1,
          xref: "x",
          yref: "paper",
          line: { color: "rgba(10, 28, 36, 0.4)", width: 1, dash: "dot" },
        },
      ]
    : [];

  const layout = {
    margin: { l: 40, r: 20, t: 10, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    showlegend: false,
    xaxis: { title: "Year" },
    yaxis: { title: label },
    font: { family: "Space Grotesk" },
    shapes,
  };

  try {
    Plotly.react(el, [trace, forecastTrace], layout, { displayModeBar: false, responsive: true });
  } catch (err) {
    console.error(`[ClimateLens] Chart render failed: ${containerId}`, err);
    return false;
  }
  return true;
}

function App() {
  const [mode, setMode] = useState("simple");
  const [query, setQuery] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [charts, setCharts] = useState(null);
  const [timelineIndex, setTimelineIndex] = useState(0);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [compareValue, setCompareValue] = useState(50);
  const [errorMessage, setErrorMessage] = useState("");
  const [noticeMessage, setNoticeMessage] = useState("");

  const mapRef = useRef(null);
  const overlayRef = useRef({});
  const layerControlRef = useRef(null);
  const baseLayersRef = useRef(null);

  const isExpert = mode === "expert";

  useEffect(() => {
    const map = L.map("map", { zoomControl: false }).setView([37.7749, -122.4194], 9);
    L.control.zoom({ position: "bottomright" }).addTo(map);

    const streets = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "© OpenStreetMap",
    });
    const satellite = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { attribution: "Tiles © Esri" }
    );
    streets.addTo(map);

    baseLayersRef.current = { Streets: streets, "Satellite (Esri)": satellite };
    layerControlRef.current = L.control.layers(baseLayersRef.current, {}, { collapsed: false }).addTo(map);

    mapRef.current = map;
  }, []);

  useEffect(() => {
    if (!analysis) {
      return;
    }
    const timelineYears = analysis.timeline.years;
    const endYear = analysis.years[analysis.years.length - 1];
    const idx = timelineYears.indexOf(endYear);
    setTimelineIndex(idx >= 0 ? idx : 0);
  }, [analysis]);

  useEffect(() => {
    if (!analysis) {
      return;
    }
    const year = analysis.timeline.years[timelineIndex];
    const overlays = analysis.timeline.overlays[String(year)] || {};
    const ndviOverlay = analysis.timeline.ndvi ? analysis.timeline.ndvi[String(year)] : null;
    const merged = { ...overlays };
    if (ndviOverlay) {
      merged.ndvi = ndviOverlay;
    }
    applyOverlays(merged, analysis.bbox);
  }, [analysis, timelineIndex, mode]);

  useEffect(() => {
    if (!charts || !analysis) {
      return;
    }
    if (!analysis.timeline || !Array.isArray(analysis.timeline.years)) {
      logDebug("Chart render skipped (timeline missing)");
      return;
    }
    const year = analysis.timeline.years[timelineIndex];
    if (!year) {
      logDebug("Chart render skipped (missing year)", { timelineIndex });
      return;
    }

    logDebug("Rendering charts", {
      mode,
      year,
      hasCharts: true,
      years: charts.years ? charts.years.length : 0,
      location: analysis.location ? analysis.location.name : "--",
    });

    safeRenderChart({
      containerId: "chart-veg",
      label: "Vegetation %",
      years: charts.years,
      series: charts.series ? charts.series.vegetation : null,
      forecastYears: charts.forecast_years,
      forecast: charts.forecast ? charts.forecast.vegetation : null,
      color: COLORS.vegetation,
      markerYear: year,
      mode,
    });

    if (isExpert) {
      safeRenderChart({
        containerId: "chart-water",
        label: "Water %",
        years: charts.years,
        series: charts.series ? charts.series.water : null,
        forecastYears: charts.forecast_years,
        forecast: charts.forecast ? charts.forecast.water : null,
        color: COLORS.water,
        markerYear: year,
        mode,
      });
    }

    safeRenderChart({
      containerId: "chart-urban",
      label: "Urban %",
      years: charts.years,
      series: charts.series ? charts.series.urban : null,
      forecastYears: charts.forecast_years,
      forecast: charts.forecast ? charts.forecast.urban : null,
      color: COLORS.urban,
      markerYear: year,
      mode,
    });
  }, [charts, analysis, timelineIndex, mode, isExpert]);

  function applyOverlays(overlays, bbox) {
    const map = mapRef.current;
    if (!map || !bbox) {
      return;
    }
    const bounds = L.latLngBounds([bbox.south, bbox.west], [bbox.north, bbox.east]);

    Object.values(overlayRef.current).forEach((layer) => map.removeLayer(layer));
    overlayRef.current = {};

    if (layerControlRef.current) {
      layerControlRef.current.remove();
    }
    layerControlRef.current = L.control.layers(baseLayersRef.current, {}, { collapsed: false }).addTo(map);

    const labels = {
      vegetation: "Vegetation change",
      urban: "Urban growth",
      water: "Water change",
      ice: "Ice change",
      ndvi: "NDVI heatmap",
    };

    const defaultVisible = isExpert ? ["ndvi", "vegetation", "urban"] : ["ndvi", "vegetation"];

    Object.entries(overlays).forEach(([key, url]) => {
      if (!url) {
        return;
      }
      const overlay = L.imageOverlay(url, bounds, { opacity: 0.65 });
      if (defaultVisible.includes(key)) {
        overlay.addTo(map);
      }
      layerControlRef.current.addOverlay(overlay, labels[key] || `${key} overlay`);
      overlayRef.current[key] = overlay;
    });

    map.fitBounds(bounds, { padding: [30, 30] });
  }

  async function analyze() {
    setLoading(true);
    setErrorMessage("");
    setNoticeMessage("");
    setAnswer("");
    logDebug("Analyze requested", { query, mode });
    try {
      const payload = {
        query: query || null,
        mode,
        year_start: DEFAULT_YEAR_START,
        year_end: DEFAULT_YEAR_END,
      };
      const analysisResponse = await postJSON("/analyze_location", payload);
      const chartsResponse = await postJSON("/get_charts_data", payload);
      setAnalysis(analysisResponse);
      setCharts(chartsResponse);
      logDebug("Analysis updated", {
        location: analysisResponse.location ? analysisResponse.location.name : "--",
        years: analysisResponse.years,
        bbox: analysisResponse.bbox,
      });
      if (analysisResponse.notices && analysisResponse.notices.length) {
        setNoticeMessage(analysisResponse.notices[0].message);
      }
    } catch (err) {
      setAnalysis(null);
      setCharts(null);
      setErrorMessage(err.message || "Unable to analyze location.");
    } finally {
      setLoading(false);
    }
  }

  async function askQuestion(customQuestion) {
    const questionText = customQuestion || question;
    if (!questionText.trim()) {
      return;
    }
    if (!analysis) {
      setAnswer("Please analyze a location first.");
      return;
    }
    setAssistantLoading(true);
    logDebug("Assistant request", {
      question: questionText,
      mode,
      year: analysis ? analysis.timeline.years[timelineIndex] : null,
    });
    try {
      const timelineYear = analysis ? analysis.timeline.years[timelineIndex] : null;
      const response = await postJSON("/ai_explanation", {
        query: query || null,
        mode,
        question: questionText,
        timeline_year: timelineYear,
        year_start: DEFAULT_YEAR_START,
        year_end: DEFAULT_YEAR_END,
        require_cached: true,
        analysis_id: analysis ? analysis.analysis_id : null,
        analysis_summary: analysis ? analysis.summary : null,
        analysis_metrics: analysis
          ? {
              changes: analysis.changes,
              indices: analysis.indices,
              years: analysis.years,
            }
          : null,
      });
      setAnswer(response.summary);
    } catch (err) {
      setAnswer(err.message || "AI assistant is unavailable.");
    } finally {
      setAssistantLoading(false);
    }
  }

  useEffect(() => {
    analyze();
  }, []);

  useEffect(() => {
    if (analysis) {
      analyze();
    }
  }, [mode]);

  const timelineYears = analysis ? analysis.timeline.years : [];
  const selectedYear = analysis ? timelineYears[timelineIndex] : null;
  const startYear = analysis ? analysis.years[0] : DEFAULT_YEAR_START;
  const endYear = analysis ? analysis.years[analysis.years.length - 1] : DEFAULT_YEAR_END;
  const forecasted = analysis && selectedYear > endYear;

  const factor = analysis && selectedYear ? computeFactor(selectedYear, startYear, endYear) : 0;
  const displayChanges = analysis
    ? {
        vegetation: analysis.changes.vegetation * factor,
        water: analysis.changes.water * factor,
        urban: analysis.changes.urban * factor,
        ice: analysis.changes.ice * factor,
      }
    : null;

  const previewBase = analysis ? analysis.timeline.previews[String(startYear)] : null;
  const previewSelected = analysis && selectedYear ? analysis.timeline.previews[String(selectedYear)] : null;

  const indicesByYear = analysis && analysis.indices && analysis.indices.by_year ? analysis.indices.by_year : {};
  const indicesYear = indicesByYear[String(selectedYear)] ? selectedYear : endYear;
  const activeIndices = indicesByYear[String(indicesYear)] || null;

  const activeSource = analysis && analysis.sources ? analysis.sources[String(endYear)] : null;
  const sourceLabel = activeSource === "sentinel" ? "Sentinel-2" : "Demo dataset";

  const comparisonLabel = analysis ? `Comparing ${startYear} -> ${endYear}` : "Comparing --";
  const selectedLabel = selectedYear ? `Selected year: ${selectedYear}` : "Selected year: --";

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <p className="eyebrow">ClimateLens</p>
          <h1>ClimateLens</h1>
          <p className="subtitle">See how Earth is changing through AI-powered satellite analysis.</p>
        </div>
        <div className="top-controls">
          <div>
            <p className="label">Mode</p>
            <div className="mode-toggle">
              <button className={mode === "simple" ? "active" : ""} onClick={() => setMode("simple")}>Simple</button>
              <button className={mode === "expert" ? "active" : ""} onClick={() => setMode("expert")}>Expert</button>
            </div>
          </div>
          <div>
            <p className="label">Location</p>
            <div className="search">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="City or lat, lon"
              />
              <button onClick={analyze} disabled={loading}>{loading ? "Analyzing..." : "Analyze"}</button>
            </div>
            <p className="helper">Examples: "San Francisco", "Bishkek", "43.2389, 76.8897"</p>
            {errorMessage && <div className="alert error">{errorMessage}</div>}
            {noticeMessage && <div className="alert notice">{noticeMessage}</div>}
          </div>
        </div>
      </header>

      <main className="dashboard">
        <aside className={`panel left-panel ${isExpert ? "expert" : "simple"}`}>
          <div className="panel-header">
            <h2>{isExpert ? "Expert Insights" : "Quick Insights"}</h2>
            <span className="pill">{sourceLabel}</span>
          </div>

          {isExpert ? (
            <div className="expert-sections">
              <div className="section">
                <h3>Key Changes</h3>
                <p className="summary">{analysis ? analysis.summary : "Run an analysis to see key changes."}</p>
              </div>
              <div className="section">
                <h3>Impact</h3>
                <p className="insight-text">{analysis && analysis.insights ? analysis.insights.impact : "--"}</p>
              </div>
              <div className="section">
                <h3>Possible Causes</h3>
                <p className="insight-text">{analysis && analysis.insights ? analysis.insights.causes : "--"}</p>
              </div>
              <div className="section">
                <h3>Recommendations</h3>
                <div className="recommendations">
                  {analysis && analysis.recommendations.map((rec) => (
                    <span key={rec}>{rec}</span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="simple-sections">
              <p className="summary">{analysis ? analysis.summary : "Analyze a location to see a plain-language summary."}</p>
              <div className="recommendations">
                {analysis && analysis.recommendations.slice(0, 1).map((rec) => (
                  <span key={rec}>{rec}</span>
                ))}
              </div>
            </div>
          )}

          <div className="indicator-grid">
            <div className="stat-card">
              <p className="label">Vegetation change</p>
              <p className="value">{displayChanges ? formatChange(displayChanges.vegetation) : "--"}</p>
            </div>
            <div className="stat-card">
              <p className="label">Urban change</p>
              <p className="value">{displayChanges ? formatChange(displayChanges.urban) : "--"}</p>
            </div>
            <div className="stat-card">
              <p className="label">Water change</p>
              <p className="value">{displayChanges ? formatChange(displayChanges.water) : "--"}</p>
            </div>
            {isExpert && (
              <>
                <div className="stat-card">
                  <p className="label">NDVI mean</p>
                  <p className="value">{activeIndices ? activeIndices.ndvi_mean.toFixed(2) : "--"}</p>
                </div>
                <div className="stat-card">
                  <p className="label">NDWI mean</p>
                  <p className="value">{activeIndices ? activeIndices.ndwi_mean.toFixed(2) : "--"}</p>
                </div>
                <div className="stat-card">
                  <p className="label">NDBI mean</p>
                  <p className="value">{activeIndices ? activeIndices.ndbi_mean.toFixed(2) : "--"}</p>
                </div>
              </>
            )}
          </div>

          <div className="assistant">
            <p className="label">Ask the assistant</p>
            <div className="prompt-row">
              {PROMPTS.map((prompt) => (
                <button key={prompt} className="prompt" onClick={() => { setQuestion(prompt); askQuestion(prompt); }}>
                  {prompt}
                </button>
              ))}
            </div>
            <div className="search">
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask about vegetation, water, urban growth, or change drivers..."
              />
              <button onClick={() => askQuestion()} disabled={assistantLoading}>
                {assistantLoading ? "Thinking..." : "Ask"}
              </button>
            </div>
            <div className="answer">{assistantLoading ? "Generating answer..." : answer}</div>
          </div>
        </aside>

        <section className="panel map-panel">
          <div className="panel-header">
            <div>
              <h2>Change Explorer</h2>
              <p className="helper">Toggle NDVI, urban growth, and water change layers.</p>
            </div>
            <div className="panel-meta">
              <span>{analysis ? analysis.location.name : "Awaiting analysis..."}</span>
              <span>{analysis ? `${startYear} -> ${selectedYear}${forecasted ? " forecast" : ""}` : "--"}</span>
            </div>
          </div>
          <div className="map-wrap">
            <div id="map" className="map"></div>
            <div className="map-legend">
              <div className="legend-row">
                <span className="legend-swatch decrease"></span>
                <span>Decrease</span>
              </div>
              <div className="legend-row">
                <span className="legend-swatch increase"></span>
                <span>Increase</span>
              </div>
              <div className="legend-row">
                <span className="legend-swatch ndvi"></span>
                <span>NDVI heatmap</span>
              </div>
            </div>
          </div>
          <div className="panel-footer">
            <span>{analysis ? `${analysis.cached ? "cached" : "computed"}` : ""}</span>
            <span>{analysis ? `Source: ${sourceLabel}` : ""}</span>
          </div>
          <div className="compare">
            <div className="compare-images">
              {previewBase && <img src={previewBase} alt="Baseline" className="compare-base" />}
              {previewSelected && (
                <div className="compare-top" style={{ width: `${compareValue}%` }}>
                  <img src={previewSelected} alt="Selected" className="compare-top-img" />
                </div>
              )}
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={compareValue}
              onChange={(e) => setCompareValue(Number(e.target.value))}
            />
            <div className="compare-labels">
              <span>{startYear}</span>
              <span>{selectedYear || "--"}</span>
            </div>
          </div>
        </section>

        <aside className={`panel right-panel ${isExpert ? "expert" : "simple"}`}>
          <div className="panel-header">
            <h2>{isExpert ? "Detailed Trends" : "Key Trends"}</h2>
          </div>
          <div className="chart-block">
            <h3>Vegetation</h3>
            <div id="chart-veg" className="chart"></div>
          </div>
          {isExpert && (
            <div className="chart-block">
              <h3>Water Levels</h3>
              <div id="chart-water" className="chart"></div>
            </div>
          )}
          <div className="chart-block">
            <h3>Urban Expansion</h3>
            <div id="chart-urban" className="chart"></div>
          </div>
          {!isExpert && (
            <p className="helper">Switch to Expert mode for full charts and diagnostics.</p>
          )}
        </aside>
      </main>

      <section className="panel timeline">
        <div className="panel-header">
          <h2>Timeline</h2>
          <span>{selectedYear ? `Year ${selectedYear}${forecasted ? " (forecast)" : ""}` : ""}</span>
        </div>
        <div className="timeline-meta">
          <span>{comparisonLabel}</span>
          <span>{selectedLabel}</span>
        </div>
        <div className="slider-wrap">
          <input
            type="range"
            min="0"
            max={Math.max(timelineYears.length - 1, 0)}
            step="1"
            value={timelineIndex}
            onChange={(e) => setTimelineIndex(Number(e.target.value))}
            disabled={!analysis}
          />
          <div className="slider-labels">
            <span>{timelineYears[0] || ""}</span>
            <span>{timelineYears[timelineYears.length - 1] || ""}</span>
          </div>
        </div>
      </section>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
