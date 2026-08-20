import axios from "axios";

// The backend base URL. Override with VITE_API_BASE if needed.
const baseURL = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const api = axios.create({ baseURL, timeout: 30000 });

export const getHealth = () => api.get("/api/health").then((r) => r.data);
// Which model bundles the backend can actually serve, with class names + reasons.
export const getBundles = () => api.get("/api/bundles").then((r) => r.data);
export const getMetrics = (version = "binary") => api.get("/api/model/metrics", { params: { version } }).then((r) => r.data);
export const getConfusion = (model, version = "binary") =>
  api.get("/api/model/confusion-matrix", { params: { model, version } }).then((r) => r.data);
export const getShap = (version = "binary") => api.get("/api/explainability/shap", { params: { version } }).then((r) => r.data);
export const getAttention = (version = "binary") =>
  api.get("/api/explainability/attention", { params: { version } }).then((r) => r.data);
export const getResearch = (version = "binary") =>
  api.get("/api/research/summary", { params: { version } }).then((r) => r.data);
export const getDiagnostics = (version = "binary") =>
  api.get("/api/model/diagnostics", { params: { version } }).then((r) => r.data);
export const getModelConfig = (version = "binary") =>
  api.get("/api/model/config", { params: { version } }).then((r) => r.data);
export const getSubjects = (version = "binary") => api.get("/api/demo/subjects", { params: { version } }).then((r) => r.data);
export const getSession = (subjectId, trialId = 0, version = "binary") =>
  api.get(`/api/demo/session/${subjectId}/${trialId}`, { params: { version } }).then((r) => r.data);
export const predict = (payload, version = "binary") =>
  api.post("/api/predict", payload, { params: { version } }).then((r) => r.data);
export const predictSession = (subjectId, trialId = 0, version = "binary") =>
  api
    .post("/api/predict/session", { subject_id: subjectId, trial_id: trialId }, { params: { version } })
    .then((r) => r.data);

export const uploadCsv = (file, version = "binary") => {
  const form = new FormData();
  form.append("file", file);
  return api
    .post("/api/predict/upload", form, {
      params: { version },
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 120000,
    })
    .then((r) => r.data);
};

export const uploadMat = (eegFile, nirsFile, version = "binary") => {
  const form = new FormData();
  form.append("eeg_file", eegFile);
  form.append("nirs_file", nirsFile);
  return api
    .post("/api/predict/upload_mat", form, {
      params: { version },
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 300000, // 5 minutes because extraction can be slow
    })
    .then((r) => r.data);
};

// Direct URL for the example-CSV download link.
export const uploadTemplateUrl = (version = "binary") => `${baseURL}/api/predict/upload/template?version=${version}`;

// ── Evaluation runs ─────────────────────────────────────────────────────────
// Every one of these returns an artifact a training run wrote to disk under
// outputs/runs/<run_id>/. The run is self-describing (it carries its own class
// names and classification mode), so nothing here assumes binary or 3-class.

export const getRuns = (params = {}) => api.get("/api/runs", { params }).then((r) => r.data);
export const getRun = (runId) => api.get(`/api/runs/${runId}`).then((r) => r.data);
export const getRunMetrics = (runId) => api.get(`/api/runs/${runId}/metrics`).then((r) => r.data);
export const getRunConfusion = (runId) =>
  api.get(`/api/runs/${runId}/confusion-matrix`).then((r) => r.data);
export const getRunCurves = (runId) => api.get(`/api/runs/${runId}/curves`).then((r) => r.data);
export const getRunHistory = (runId) =>
  api.get(`/api/runs/${runId}/training-history`).then((r) => r.data);
export const getRunPerSubject = (runId) =>
  api.get(`/api/runs/${runId}/per-subject`).then((r) => r.data);
export const getRunFeatureImportance = (runId) =>
  api.get(`/api/runs/${runId}/feature-importance`).then((r) => r.data);
export const getRunRepresentations = (runId) =>
  api.get(`/api/runs/${runId}/representations`).then((r) => r.data);
export const getRunExperiments = (runId) =>
  api.get(`/api/runs/${runId}/experiments`).then((r) => r.data);
export const getRunPredictions = (runId, params = {}) =>
  api.get(`/api/runs/${runId}/predictions`, { params }).then((r) => r.data);

// Direct URL for a figure the run rendered (served as image/png).
export const runFigureUrl = (runId, path) =>
  `${baseURL}/api/runs/${runId}/figure?path=${encodeURIComponent(path)}`;

export default api;
