import axios from "axios";

// The backend base URL. Override with VITE_API_BASE if needed.
const baseURL = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const api = axios.create({ baseURL, timeout: 30000 });

export const getHealth = <T = any,>(): Promise<T> => api.get("/api/health").then((r) => r.data);
// Which model bundles the backend can actually serve, with class names + reasons.
export const getBundles = <T = any,>(): Promise<T> => api.get("/api/bundles").then((r) => r.data);
export const getMetrics = <T = any,>(version: string = "binary"): Promise<T> => api.get("/api/model/metrics", { params: { version } }).then((r) => r.data);
export const getConfusion = <T = any,>(model: string, version: string = "binary"): Promise<T> =>
  api.get("/api/model/confusion-matrix", { params: { model, version } }).then((r) => r.data);
export const getShap = <T = any,>(version: string = "binary"): Promise<T> => api.get("/api/explainability/shap", { params: { version } }).then((r) => r.data);
export const getAttention = <T = any,>(version: string = "binary"): Promise<T> =>
  api.get("/api/explainability/attention", { params: { version } }).then((r) => r.data);
export const getResearch = <T = any,>(version: string = "binary"): Promise<T> =>
  api.get("/api/research/summary", { params: { version } }).then((r) => r.data);
export const getDiagnostics = <T = any,>(version: string = "binary"): Promise<T> =>
  api.get("/api/model/diagnostics", { params: { version } }).then((r) => r.data);
export const getModelConfig = <T = any,>(version: string = "binary"): Promise<T> =>
  api.get("/api/model/config", { params: { version } }).then((r) => r.data);
export const getSubjects = <T = any,>(version: string = "binary"): Promise<T> => api.get("/api/demo/subjects", { params: { version } }).then((r) => r.data);
export const getSession = <T = any,>(subjectId: number | string, trialId: number = 0, version: string = "binary"): Promise<T> =>
  api.get(`/api/demo/session/${subjectId}/${trialId}`, { params: { version } }).then((r) => r.data);
export const predict = <T = any,>(payload: unknown, version: string = "binary"): Promise<T> =>
  api.post("/api/predict", payload, { params: { version } }).then((r) => r.data);
export const predictSession = <T = any,>(subjectId: number | string, trialId: number = 0, version: string = "binary"): Promise<T> =>
  api
    .post("/api/predict/session", { subject_id: subjectId, trial_id: trialId }, { params: { version } })
    .then((r) => r.data);

export const uploadCsv = <T = any,>(file: File, version: string = "binary"): Promise<T> => {
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

export const uploadMat = <T = any,>(eegFile: File, nirsFile: File, version: string = "binary"): Promise<T> => {
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

export const getUniversalDatasets = <T = any,>(): Promise<T> =>
  api.get("/api/predict/universal/datasets").then((r) => r.data);

export const uploadUniversal = <T = any,>(files: File[], datasetProfile: string = "", version: string = "binary"): Promise<T> => {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }
  const params: Record<string, string> = { version };
  if (datasetProfile) params.dataset = datasetProfile;
  return api
    .post("/api/predict/universal", form, {
      params,
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 300000,
    })
    .then((r) => r.data);
};

// Direct URL for the example-CSV download link.
export const uploadTemplateUrl = (version: string = "binary"): string => `${baseURL}/api/predict/upload/template?version=${version}`;

// ── Evaluation runs ─────────────────────────────────────────────────────────
// Every one of these returns an artifact a training run wrote to disk under
// outputs/runs/<run_id>/. The run is self-describing (it carries its own class
// names and classification mode), so nothing here assumes binary or 3-class.

export const getRuns = <T = any,>(params: Record<string, unknown> = {}): Promise<T> => api.get("/api/runs", { params }).then((r) => r.data);
export const getRun = <T = any,>(runId: string): Promise<T> => api.get(`/api/runs/${runId}`).then((r) => r.data);
export const getRunMetrics = <T = any,>(runId: string): Promise<T> => api.get(`/api/runs/${runId}/metrics`).then((r) => r.data);
export const getRunConfusion = <T = any,>(runId: string): Promise<T> =>
  api.get(`/api/runs/${runId}/confusion-matrix`).then((r) => r.data);
export const getRunCurves = <T = any,>(runId: string): Promise<T> => api.get(`/api/runs/${runId}/curves`).then((r) => r.data);
export const getRunHistory = <T = any,>(runId: string): Promise<T> =>
  api.get(`/api/runs/${runId}/training-history`).then((r) => r.data);
export const getRunPerSubject = <T = any,>(
  runId: string,
  experiment?: string
): Promise<T> =>
  api
    .get(`/api/runs/${runId}/per-subject`, {
      params: experiment ? { experiment } : undefined,
    })
    .then((r) => r.data);
export const getRunFeatureImportance = <T = any,>(runId: string): Promise<T> =>
  api.get(`/api/runs/${runId}/feature-importance`).then((r) => r.data);
export const getRunRepresentations = <T = any,>(runId: string): Promise<T> =>
  api.get(`/api/runs/${runId}/representations`).then((r) => r.data);
export const getRunExperiments = <T = any,>(runId: string): Promise<T> =>
  api.get(`/api/runs/${runId}/experiments`).then((r) => r.data);
export const getRunSummary = <T = any,>(runId: string): Promise<T> =>
  api.get(`/api/runs/${runId}/summary`).then((r) => r.data);
export const getRunPredictions = <T = any,>(runId: string, params: Record<string, unknown> = {}): Promise<T> =>
  api.get(`/api/runs/${runId}/predictions`, { params }).then((r) => r.data);

// Direct URL for a figure the run rendered (served as image/png).
export const runFigureUrl = (runId: string, path: string): string =>
  `${baseURL}/api/runs/${runId}/figure?path=${encodeURIComponent(path)}`;

export default api;
