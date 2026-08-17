import axios from "axios";

// The backend base URL. Override with VITE_API_BASE if needed.
const baseURL = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const api = axios.create({ baseURL, timeout: 30000 });

export const getHealth = () => api.get("/api/health").then((r) => r.data);
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

export default api;
