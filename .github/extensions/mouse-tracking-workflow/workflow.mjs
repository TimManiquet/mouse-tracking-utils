import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { CanvasError } from "@github/copilot-sdk/extension";

const FORMAT = "mouse-tracking-utils-workflow";
const VERSION = 1;
const htmlPath = fileURLToPath(new URL("./canvas.html", import.meta.url));
const liveInstances = new Map();

const initialMap = () => ({
  format: FORMAT,
  version: VERSION,
  nodes: [
    { id: "raw-data", label: "Raw CSV / DataFrame", details: "One row per trial; one file per participant", category: "input" },
    { id: "check-columns", label: "check_columns", details: "Validate or map input column names", category: "utility" },
    { id: "preprocess", label: "preprocess_mouse_tracking_data", details: "Filter rows; standardize trajectories and button positions", category: "utility" },
    { id: "filter", label: "filter_df", details: "Apply row filters to a dataframe", category: "utility" },
    { id: "exclude-trials", label: "exclude_trials", details: "Exclude trials by endpoint or travelled distance", category: "utility" },
    { id: "extract-distance", label: "extract_distance", details: "Interpolate trajectories; calculate distances or positions", category: "utility" },
    { id: "movement-data", label: "MotorMovementData", details: "Named dimensions, coordinates, and metadata", category: "data" },
    { id: "average", label: "average", details: "NaN-aware mean along named dimensions", category: "utility" },
    { id: "downsample", label: "downsample", details: "Reduce time sampling frequency", category: "utility" },
    { id: "save-load", label: "save / load", details: "Persist the data object as .npz", category: "output" },
    { id: "plotting", label: "plot_trials / plot_binary_trials", details: "Plot trajectories and screen layout", category: "output" },
    { id: "demo-data", label: "demo data (2 / 4 / 8)", details: "Generate or load synthetic example trials", category: "input" },
  ],
  edges: [
    { from: "raw-data", to: "check-columns", label: "optional preflight" },
    { from: "check-columns", to: "preprocess" },
    { from: "raw-data", to: "preprocess" },
    { from: "preprocess", to: "filter" },
    { from: "preprocess", to: "exclude-trials" },
    { from: "preprocess", to: "extract-distance" },
    { from: "filter", to: "extract-distance" },
    { from: "exclude-trials", to: "extract-distance" },
    { from: "preprocess", to: "plotting" },
    { from: "demo-data", to: "preprocess" },
    { from: "extract-distance", to: "movement-data" },
    { from: "movement-data", to: "average" },
    { from: "movement-data", to: "downsample" },
    { from: "movement-data", to: "save-load" },
    { from: "downsample", to: "save-load" },
  ],
});

function validateMap(data) {
  if (
    !data ||
    data.format !== FORMAT ||
    data.version !== VERSION ||
    !Array.isArray(data.nodes) ||
    !Array.isArray(data.edges)
  ) {
    throw new CanvasError(
      "workflow_data_invalid",
      "The workflow map has an unsupported format or invalid shape. It was left unchanged.",
    );
  }
  const ids = new Set();
  for (const node of data.nodes) {
    if (
      !node ||
      typeof node.id !== "string" ||
      !/^[a-z0-9][a-z0-9-]{0,79}$/.test(node.id) ||
      typeof node.label !== "string" ||
      typeof node.details !== "string" ||
      !["input", "utility", "data", "output"].includes(node.category) ||
      ids.has(node.id)
    ) {
      throw new CanvasError("workflow_data_invalid", "A workflow node is invalid or duplicated. The file was left unchanged.");
    }
    ids.add(node.id);
  }
  for (const edge of data.edges) {
    if (
      !edge ||
      typeof edge.from !== "string" ||
      typeof edge.to !== "string" ||
      !ids.has(edge.from) ||
      !ids.has(edge.to) ||
      typeof edge.label !== "string"
    ) {
      throw new CanvasError("workflow_data_invalid", "A workflow dependency is invalid. The file was left unchanged.");
    }
  }
  return data;
}

async function saveMap(filePath, data) {
  validateMap(data);
  const tempPath = join(dirname(filePath), `.workflow-map-${randomUUID()}.tmp`);
  try {
    await writeFile(tempPath, `${JSON.stringify(data, null, 2)}\n`, { flag: "wx" });
    await rename(tempPath, filePath);
  } catch (error) {
    throw new CanvasError("workflow_storage_failed", `Could not save workflow map: ${error.message}`);
  }
}

async function loadMap(filePath) {
  try {
    const text = await readFile(filePath, "utf8");
    return validateMap(JSON.parse(text));
  } catch (error) {
    if (error.code !== "ENOENT") {
      if (error instanceof CanvasError) throw error;
      throw new CanvasError("workflow_storage_failed", `Could not read workflow map: ${error.message}`);
    }
    const data = initialMap();
    await saveMap(filePath, data);
    return data;
  }
}

function slugify(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "utility";
}

function validateText(value, field, maxLength, optional = false) {
  if (optional && value === undefined) return "";
  if (typeof value !== "string" || value.length > maxLength || (!optional && !value.trim())) {
    throw new CanvasError("invalid_input", `${field} must be a non-empty string of at most ${maxLength} characters.`);
  }
  return value.trim();
}

function hasPath(edges, from, to) {
  const stack = [from];
  const seen = new Set();
  while (stack.length) {
    const current = stack.pop();
    if (current === to) return true;
    if (seen.has(current)) continue;
    seen.add(current);
    for (const edge of edges) if (edge.from === current) stack.push(edge.to);
  }
  return false;
}

async function readRequestBody(request) {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
    if (body.length > 16_384) throw new CanvasError("invalid_request", "Request body is too large.");
  }
  try {
    return JSON.parse(body);
  } catch {
    throw new CanvasError("invalid_request", "Request body must be valid JSON.");
  }
}

function sendJson(response, statusCode, value) {
  response.writeHead(statusCode, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
  response.end(JSON.stringify(value));
}

function sendError(response, error) {
  const status = error instanceof CanvasError && error.code === "invalid_input" ? 400 : 500;
  sendJson(response, status, { error: error.message || "Workflow request failed." });
}

async function handleRequest(instance, request, response) {
  try {
    const url = new URL(request.url, "http://127.0.0.1");
    if (request.method === "GET" && url.pathname === "/") {
      const html = await readFile(htmlPath);
      response.writeHead(200, {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
        "content-security-policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self';",
        "x-content-type-options": "nosniff",
      });
      response.end(html);
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/workflow") {
      sendJson(response, 200, await loadMap(instance.filePath));
      return;
    }
    if (request.method !== "POST") {
      sendJson(response, 404, { error: "Not found." });
      return;
    }
    const input = await readRequestBody(request);
    if (url.pathname === "/api/utilities") {
      sendJson(response, 200, await addUtility(instance, input));
      return;
    }
    if (url.pathname === "/api/dependencies") {
      sendJson(response, 200, await connectUtilities(instance, input));
      return;
    }
    sendJson(response, 404, { error: "Not found." });
  } catch (error) {
    sendError(response, error);
  }
}

export async function openWorkflowCanvas(instanceId, workingDirectory) {
  if (typeof instanceId !== "string" || !instanceId) {
    throw new CanvasError("invalid_instance", "A canvas instance identifier is required.");
  }
  if (typeof workingDirectory !== "string" || !workingDirectory) {
    throw new CanvasError("workspace_unknown", "The active workspace path is unavailable.");
  }
  const filePath = resolve(workingDirectory, "docs", "workflow-map.json");
  await mkdir(dirname(filePath), { recursive: true });
  await loadMap(filePath);
  const existing = liveInstances.get(instanceId);
  if (existing && existing.filePath === filePath && existing.server.listening) {
    await loadMap(filePath);
    const address = existing.server.address();
    return { ...existing, url: `http://127.0.0.1:${address.port}/` };
  }
  if (existing) await closeServer(existing.server);

  const instance = { filePath, server: createServer((req, res) => void handleRequest({ filePath }, req, res)) };
  await new Promise((resolveListen, reject) => {
    instance.server.once("error", reject);
    instance.server.listen(0, "127.0.0.1", resolveListen);
  });
  liveInstances.set(instanceId, instance);
  const address = instance.server.address();
  return { ...instance, url: `http://127.0.0.1:${address.port}/` };
}

function closeServer(server) {
  return new Promise((resolveClose) => {
    if (!server.listening) return resolveClose();
    server.close(() => resolveClose());
  });
}

export function stopWorkflowCanvas(instanceId) {
  const instance = liveInstances.get(instanceId);
  if (!instance) return;
  liveInstances.delete(instanceId);
  void closeServer(instance.server);
}

export async function getWorkflow(instance) {
  return loadMap(instance.filePath);
}

export async function addUtility(instance, input) {
  const label = validateText(input?.label, "label", 80);
  const details = validateText(input?.details, "details", 240, true);
  const category = input?.category ?? "utility";
  if (!["input", "utility", "data", "output"].includes(category)) {
    throw new CanvasError("invalid_input", "category must be input, utility, data, or output.");
  }
  const data = await loadMap(instance.filePath);
  const baseId = slugify(label);
  let id = baseId;
  let suffix = 2;
  while (data.nodes.some((node) => node.id === id)) id = `${baseId}-${suffix++}`;
  data.nodes.push({ id, label, details, category });
  await saveMap(instance.filePath, data);
  return data;
}

export async function connectUtilities(instance, input) {
  const from = validateText(input?.from, "from", 80);
  const to = validateText(input?.to, "to", 80);
  const label = validateText(input?.label, "label", 80, true);
  const data = await loadMap(instance.filePath);
  if (from === to || !data.nodes.some((node) => node.id === from) || !data.nodes.some((node) => node.id === to)) {
    throw new CanvasError("invalid_input", "from and to must be different IDs of existing workflow nodes.");
  }
  if (data.edges.some((edge) => edge.from === from && edge.to === to)) {
    throw new CanvasError("duplicate_dependency", `Dependency ${from} → ${to} already exists.`);
  }
  if (hasPath(data.edges, to, from)) {
    throw new CanvasError("cyclic_dependency", "This dependency would create a cycle.");
  }
  data.edges.push({ from, to, label });
  await saveMap(instance.filePath, data);
  return data;
}
