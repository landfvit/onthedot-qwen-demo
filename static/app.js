"use strict";

const byId = (id) => document.getElementById(id);
const ui = Object.fromEntries([
  "question-form", "mode-select", "example-select", "question-input", "run-button",
  "mode-description", "request-status", "validation-label", "query-message",
  "query-output", "raw-details", "raw-output", "generation-details", "generation-mode",
  "inference-time", "result-label", "result-message", "result-content", "result-metric",
  "result-table-head", "result-table-body", "result-chart", "result-notice",
  "model-label", "model-id", "model-revision", "live-status", "processing-time",
  "catalog-details", "catalog-list",
  "sql-message", "sql-output",
  "data-label", "data-status",
].map((id) => [id.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase()), byId(id)]));

let examples = [];
let busy = false;
let ready = false;
let dataAvailable = false;
let requestNumber = 0;
let activeRequest = null;

function status(message, error = false) {
  ui.requestStatus.textContent = message;
  ui.requestStatus.classList.toggle("error", error);
}

function setBusy(value) {
  busy = value;
  ui.runButton.disabled = value || !ready || !ui.questionInput.value.trim();
  ui.runButton.textContent = value ? "Running" : "Run question";
  ui.modeSelect.disabled = value || !ready;
  ui.exampleSelect.disabled = value || !ready || !examples.length;
  ui.questionInput.disabled = value || !ready;
  ui.questionForm.setAttribute("aria-busy", String(value));
}

function setStateLabel(element, text, state = "") {
  element.textContent = text;
  element.className = `state-label${state ? ` ${state}` : ""}`;
}

function clearOutput() {
  setStateLabel(ui.validationLabel, "Waiting");
  setStateLabel(ui.resultLabel, "Waiting");
  ui.queryMessage.hidden = false;
  ui.queryMessage.textContent = "Run a question to inspect the validated query.";
  ui.queryOutput.hidden = true;
  ui.queryOutput.textContent = "";
  ui.rawDetails.hidden = true;
  ui.rawDetails.open = false;
  ui.rawOutput.textContent = "";
  ui.generationDetails.hidden = true;
  ui.resultMessage.hidden = false;
  ui.resultMessage.textContent = dataAvailable
    ? "The validated query runs against the local season dataset."
    : "No season dataset is loaded. Qwen parsing is still available.";
  ui.resultContent.hidden = true;
  ui.resultTableHead.replaceChildren();
  ui.resultTableBody.replaceChildren();
  ui.resultChart.replaceChildren();
  ui.resultNotice.hidden = true;
  ui.processingTime.textContent = "";
  ui.sqlMessage.hidden = false;
  ui.sqlMessage.textContent = "No analytics query was executed.";
  ui.sqlOutput.hidden = true;
  ui.sqlOutput.replaceChildren();
}

function describeMode() {
  const recorded = ui.modeSelect.value === "recorded";
  ui.modeDescription.textContent = recorded
    ? "Replays a saved Qwen response for an included question."
    : "Sends this question to the local Qwen model.";
  status(recorded ? "Choose an example, then run the question." : "Type your own question, then run live inference.");
}

function humanize(value) {
  return String(value ?? "").replace(/_/g, " ");
}

function formatNumber(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat("en", { maximumFractionDigits: 3 }).format(value)
    : String(value ?? "—");
}

function rowDataStatus(row) {
  if (row.data_status === "insufficient_data" || row.value == null) return "Unknown";
  if (row.data_status === "partial" || (row.known_matches != null && row.known_matches < row.matches)) return "Partial total";
  return "Complete";
}

function duration(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${formatNumber(value)} ms`;
}

function addCell(row, tag, value) {
  const cell = document.createElement(tag);
  cell.textContent = value;
  if (tag === "th") cell.scope = "col";
  row.append(cell);
}

function svgElement(tag, attributes = {}, text = null) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (text !== null) element.textContent = text;
  return element;
}

// draw only values returned by the analytics engine.
function drawChart(rows, metric, unit) {
  const numericRows = rows.filter((row) => typeof row.value === "number" && Number.isFinite(row.value));
  if (numericRows.length < 2 || numericRows.length > 12 || numericRows.length !== rows.length) return;

  const width = 440;
  const rowHeight = 29;
  const height = rowHeight * rows.length + 15;
  const labelWidth = 145;
  const plotWidth = width - labelWidth - 55;
  const min = Math.min(0, ...rows.map((row) => row.value));
  const max = Math.max(0, ...rows.map((row) => row.value));
  const span = max - min || 1;
  const x = (value) => labelWidth + ((value - min) / span) * plotWidth;
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${humanize(metric)} comparison${unit ? ` in ${humanize(unit)}` : ""}`,
  });
  svg.append(svgElement("line", { x1: x(0), x2: x(0), y1: 2, y2: height - 5, stroke: "#d9dee7" }));

  rows.forEach((row, index) => {
    const y = index * rowHeight + 7;
    const label = String(row.label ?? "");
    const displayedLabel = label.length > 22 ? `${label.slice(0, 20)}…` : label;
    const labelElement = svgElement("text", { x: 0, y: y + 13, fill: "#687281", "font-size": 11 }, displayedLabel);
    labelElement.append(svgElement("title", {}, label));
    svg.append(labelElement);
    const bar = svgElement("rect", {
      x: Math.min(x(0), x(row.value)), y, width: Math.max(Math.abs(x(row.value) - x(0)), row.value === 0 ? 0 : 1),
      height: 17, rx: 1, fill: "#3d65ba",
    });
    bar.append(svgElement("title", {}, `${label}: ${formatNumber(row.value)}${unit ? ` ${humanize(unit)}` : ""}`));
    svg.append(bar);
    svg.append(svgElement("text", { x: width, y: y + 13, fill: "#3a4759", "font-size": 11, "text-anchor": "end" }, formatNumber(row.value)));
  });
  ui.resultChart.append(svg);
}

function renderResult(result, operation) {
  if (!result) {
    setStateLabel(ui.resultLabel, "Not run");
    ui.resultMessage.textContent = "No analytics query was executed.";
    return;
  }

  const ok = result.status === "ok";
  setStateLabel(ui.resultLabel, ok ? "Complete" : humanize(result.status || "Not run"), ok ? "valid" : result.status === "data_required" ? "warning" : "");
  if (result.notice) {
    ui.resultNotice.textContent = result.notice;
    ui.resultNotice.hidden = false;
  }
  if (!ok) {
    ui.resultMessage.textContent = result.message || "This question needs clarification before it can run.";
    return;
  }

  const rows = Array.isArray(result.rows) ? result.rows : [];
  if (!rows.length) {
    ui.resultMessage.textContent = result.message || "No matching rows in the season dataset.";
    return;
  }
  ui.resultMessage.hidden = true;
  ui.resultContent.hidden = false;
  ui.resultMetric.textContent = `${humanize(result.metric) || "Result"}${result.unit ? ` (${humanize(result.unit)})` : ""}`;

  const incomplete = rows.some((row) => rowDataStatus(row) !== "Complete");
  if (incomplete) setStateLabel(ui.resultLabel, "Partial data", "warning");
  const includeMatches = rows.some((row) => row.matches != null);
  const includeMinutes = rows.some((row) => row.minutes != null);
  const includeCoverage = incomplete && rows.some((row) => row.known_matches != null);
  const labelHeading = operation === "player_trend" ? "Month"
    : operation === "team_metric" ? "Team"
      : operation?.startsWith("player") ? "Player" : "Label";
  [labelHeading, "Value", ...(incomplete ? ["Data status"] : []), ...(includeMatches ? ["Matches"] : []), ...(includeCoverage ? ["Known"] : []), ...(includeMinutes ? ["Minutes"] : [])]
    .forEach((heading) => addCell(ui.resultTableHead, "th", heading));
  for (const item of rows) {
    const row = document.createElement("tr");
    addCell(row, "td", String(item.label ?? ""));
    addCell(row, "td", formatNumber(item.value));
    if (incomplete) addCell(row, "td", rowDataStatus(item));
    if (includeMatches) addCell(row, "td", formatNumber(item.matches));
    if (includeCoverage) addCell(row, "td", formatNumber(item.known_matches));
    if (includeMinutes) addCell(row, "td", formatNumber(item.minutes));
    ui.resultTableBody.append(row);
  }
  // incomplete totals are not drawn as if they were complete comparisons.
  if (!incomplete) drawChart(rows, result.metric, result.unit);
  if (incomplete) {
    const coverageNotice = "Known counts matches with a recorded statistic. Partial totals sum known values. Rates with missing data stay unknown (—).";
    ui.resultNotice.textContent = [result.notice, coverageNotice].filter(Boolean).join(". ");
    ui.resultNotice.hidden = false;
  }
}

// show only statements and bound values returned by the backend.
function renderSql(statements) {
  const queries = Array.isArray(statements)
    ? statements.filter((query) => typeof query?.statement === "string" && query.statement.trim())
    : [];
  if (!queries.length) return;

  ui.sqlMessage.hidden = true;
  ui.sqlOutput.hidden = false;
  for (const [index, query] of queries.entries()) {
    const entry = document.createElement("article");
    entry.className = "sql-entry";
    if (queries.length > 1) {
      const heading = document.createElement("h3");
      heading.textContent = `Query ${index + 1}`;
      entry.append(heading);
    }
    const statement = document.createElement("pre");
    statement.className = "json-output sql-statement";
    statement.tabIndex = 0;
    statement.setAttribute("aria-label", `SQL statement ${index + 1}`);
    statement.textContent = query.statement;

    const parametersLabel = document.createElement("p");
    parametersLabel.className = "sql-parameters-label";
    parametersLabel.textContent = "Bound parameters";
    const parameters = document.createElement("pre");
    parameters.className = "json-output sql-parameters";
    parameters.tabIndex = 0;
    parameters.setAttribute("aria-label", `Bound parameters for statement ${index + 1}`);
    parameters.textContent = JSON.stringify(query.parameters, null, 2) ?? "Not supplied";
    entry.append(statement, parametersLabel, parameters);
    ui.sqlOutput.append(entry);
  }
}

function renderResponse(data) {
  const valid = data.validation?.ok === true;
  const query = data.query;
  setStateLabel(ui.validationLabel, valid ? "Valid" : "Invalid", valid ? "valid" : "invalid");
  ui.queryMessage.hidden = valid && query != null;
  if (!ui.queryMessage.hidden) {
    ui.queryMessage.textContent = data.validation?.error || "The model response could not be validated.";
  }
  if (query != null) {
    ui.queryOutput.textContent = JSON.stringify(query, null, 2);
    ui.queryOutput.hidden = false;
  }

  const generation = data.generation || {};
  const recorded = generation.mode === "recorded";
  ui.rawOutput.textContent = typeof generation.raw_text === "string" ? generation.raw_text : "";
  ui.rawDetails.hidden = !ui.rawOutput.textContent;
  ui.rawDetails.open = !valid && Boolean(ui.rawOutput.textContent);
  ui.generationMode.textContent = recorded ? "Recorded Qwen output" : "Live Qwen";
  const inference = duration(generation.inference_ms);
  ui.inferenceTime.textContent = inference ? `${recorded ? "Recorded inference" : "Inference"}: ${inference}` : "";
  ui.generationDetails.hidden = false;
  if (generation.model_id) ui.modelId.textContent = generation.model_id;
  if (generation.model_revision) ui.modelRevision.textContent = generation.model_revision;

  renderResult(data.result, query?.operation);
  renderSql(data.sql);
  const validation = duration(data.timing?.validation_ms);
  const analytics = duration(data.timing?.analytics_ms);
  ui.processingTime.textContent = [validation ? `Validation: ${validation}` : "", analytics ? `Analytics: ${analytics}` : ""].filter(Boolean).join(" | ");
  status(valid ? "Question parsed and validated." : "The model response did not pass validation.", !valid);
}

async function readJson(response) {
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("The server returned an unreadable response. Check the terminal and try again.");
  }
  if (!response.ok) {
    throw new Error(data.error?.message || data.detail || `Request failed (${response.status}).`);
  }
  return data;
}

ui.questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = ui.questionInput.value.trim();
  if (!question || busy) return;

  const currentRequest = ++requestNumber;
  activeRequest?.abort();
  activeRequest = new AbortController();
  clearOutput();
  setBusy(true);
  status(ui.modeSelect.value === "recorded" ? "Loading the recorded Qwen response." : "Qwen is parsing the question. The first run can take longer while the model loads.");
  try {
    const response = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: ui.modeSelect.value }), signal: activeRequest.signal,
    });
    const data = await readJson(response);
    if (currentRequest === requestNumber) renderResponse(data);
  } catch (error) {
    if (error.name !== "AbortError" && currentRequest === requestNumber) {
      status(error.message || "The request failed. Check the terminal and try again.", true);
      setStateLabel(ui.validationLabel, "Not run");
      setStateLabel(ui.resultLabel, "Not run");
    }
  } finally {
    if (currentRequest === requestNumber) setBusy(false);
  }
});

ui.modeSelect.addEventListener("change", () => {
  clearOutput();
  describeMode();
});

ui.exampleSelect.addEventListener("change", () => {
  const example = examples.find((item) => item.id === ui.exampleSelect.value);
  ui.questionInput.value = example?.question || "";
  clearOutput();
  describeMode();
  setBusy(false);
});

ui.questionInput.addEventListener("input", () => {
  clearOutput();
  ui.runButton.disabled = !ui.questionInput.value.trim();
  const example = examples.find((item) => item.question === ui.questionInput.value.trim());
  ui.exampleSelect.value = example?.id || "";
  status(ui.modeSelect.value === "recorded" && !example
    ? "Recorded mode supports included questions. Choose an example or switch to Live Qwen."
    : "Ready to run.");
});

function renderCatalog(catalog) {
  const fields = [
    ["Players", catalog?.players], ["Teams", catalog?.teams], ["Seasons", catalog?.seasons],
    ["Player statistics", catalog?.metrics?.player], ["Team statistics", catalog?.metrics?.team],
    ["Competition", catalog?.competition],
  ];
  ui.catalogList.replaceChildren();
  for (const [label, value] of fields) {
    if (value == null || (Array.isArray(value) && !value.length)) continue;
    const entry = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = Array.isArray(value) ? value.map(humanize).join(", ") : String(value);
    entry.append(term, description);
    ui.catalogList.append(entry);
  }
  ui.catalogDetails.hidden = !ui.catalogList.childElementCount;
}

// metadata controls model availability without triggering inference.
async function initialize() {
  try {
    const response = await fetch("/api/meta");
    const meta = await readJson(response);
    dataAvailable = meta.data_available === true;
    ui.dataLabel.textContent = meta.catalog?.notice || (dataAvailable ? "Local season data" : "Dataset not available");
    ui.dataStatus.textContent = "Dataset required. Qwen can still parse your question. Analytics results need the local season data.";
    ui.dataStatus.hidden = dataAvailable;
    clearOutput();
    examples = Array.isArray(meta.examples) ? meta.examples : [];
    ui.exampleSelect.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Type your own question";
    ui.exampleSelect.append(placeholder);
    for (const example of examples) {
      const option = document.createElement("option");
      option.value = example.id;
      option.textContent = example.question;
      ui.exampleSelect.append(option);
    }
    const sample = examples.find((example) => example.question.startsWith("Compare ")) || examples[0];
    ui.questionInput.placeholder = sample ? `For example: ${sample.question}` : "Ask a football statistics question.";
    if (meta.live_available) {
      ui.modeSelect.value = "live";
      ui.exampleSelect.value = "";
      ui.questionInput.value = "";
    } else if (examples.length) {
      ui.exampleSelect.value = examples[0].id;
      ui.questionInput.value = examples[0].question;
    }
    ui.modelId.textContent = meta.model_id || "Not specified";
    ui.modelRevision.textContent = meta.model_revision || "Not specified";
    ui.modelLabel.textContent = meta.model_id ? `Model: ${meta.model_id}` : "Model details";
    ui.liveStatus.textContent = meta.live_available ? "Available locally" : "Unavailable. See the setup guide.";
    const liveOption = ui.modeSelect.querySelector('option[value="live"]');
    liveOption.disabled = !meta.live_available;
    if (!meta.live_available) liveOption.textContent = "Live Qwen (setup required)";
    renderCatalog(meta.catalog);
    ready = true;
    setBusy(false);
    describeMode();
  } catch (error) {
    setBusy(false);
    status(error.message || "The workbench could not load. Check the terminal and reload.", true);
    ui.liveStatus.textContent = "Unable to connect";
  }
}

initialize();
