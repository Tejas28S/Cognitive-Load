import axios from "axios";

// The backend base URL. Override with VITE_API_BASE if needed.
const baseURL = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const api = axios.create({ baseURL, timeout: 30000 });

export const getHealth = () => api.get("/api/health").then((r) => r.data);
export const getMetrics = () => api.get("/api/model/metrics").then((r) => r.data);
export const getConfusion = (model) =>
  api.get("/api/model/confusion-matrix", { params: { model } }).then((r) => r.data);
export const getShap = () => api.get("/api/explainability/shap").then((r) => r.data);
export const getAttention = () =>
  api.get("/api/explainability/attention").then((r) => r.data);
export const getResearch = () =>
  api.get("/api/research/summary").then((r) => r.data);
export const getSubjects = () => api.get("/api/demo/subjects").then((r) => r.data);
export const getSession = (subjectId, trialId = 0) =>
  api.get(`/api/demo/session/${subjectId}/${trialId}`).then((r) => r.data);
export const predict = (payload) =>
  api.post("/api/predict", payload).then((r) => r.data);
export const predictSession = (subjectId, trialId = 0) =>
  api
    .post("/api/predict/session", { subject_id: subjectId, trial_id: trialId })
    .then((r) => r.data);

export const uploadCsv = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api
    .post("/api/predict/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 120000,
    })
    .then((r) => r.data);
};

// Direct URL for the example-CSV download link.
export const uploadTemplateUrl = `${baseURL}/api/predict/upload/template`;

export default api;
