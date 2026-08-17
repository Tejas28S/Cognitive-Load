<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 40px; color: #333; }
h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 40px; }
h3 { color: #34495e; margin-top: 30px; }
blockquote { border-left: 4px solid #e74c3c; margin: 0; padding-left: 15px; color: #555; background: #f9f9f9; padding: 10px 15px; }
.page-break { page-break-before: always; }
</style>

# Cognitive Load Estimation from EEG + fNIRS

**Reading mental workload from the brain — and reporting it honestly.**

This capstone project estimates cognitive load (Low, Medium, High) using multimodal brain signals. It uses the open TU Berlin / Shin n-back dataset.

### The True Challenge
The biggest hurdle isn't architecture—it's **subject variability**. Models easily latch onto person-specific physiological baselines that don't transfer to new people. Our primary goal is robust, cross-subject generalization.

<div class="page-break"></div>

# Architecture & Modalities

We use two distinct types of brain signals:
- **EEG (Electrical):** Fast, spiky band-power features (180 columns).
- **fNIRS (Hemodynamic):** Slower, smoother blood-oxygenation response shapes (468 columns).

### The Model
- **Hybrid CNN-LSTM:** Parallel encoders for each modality.
- **Attention-Gated Fusion:** The network learns how much to trust EEG vs. fNIRS for every window.
- **The Served Ensemble:** The final deep neural network predictions are averaged with a Random Forest model for maximum stability.

<div class="page-break"></div>

# Methodological Integrity

> **WARNING**: Literature often reports 60-90% accuracy using *fixed splits* (where the same person appears in both train and test data). This leaks subject identity and overstates real-world performance!

### The Honest Gap
We report both numbers to expose this leakage:
- **Fixed Split (Known subjects):** ~42.7% Accuracy | 0.406 Macro F1
- **LOSO (Unseen subjects):** ~39.8% Accuracy | 0.364 Macro F1

*Leave-One-Subject-Out (LOSO)* is the strict evaluation standard we use to measure how the model performs on people it has never seen. 

<div class="page-break"></div>

# Rigorous Decision Gates & DANN

Every candidate improvement (XGBoost, SVM, HbR features) faced a strict decision gate: *Only adopt if it provides a clear, statistically significant win on LOSO.*

### The DANN Experiment
We built a **Domain-Adversarial Neural Network (DANN)** to actively erase subject-specific structure from the embeddings using a gradient-reversal layer.

> **Verdict: Declined (Honest Negative Result)**
> DANN did not improve the ensemble. Why? We built a *subject-invariance probe* which proved the baseline model was already stripping out subject-identity. The adversary had nothing to remove and only added noise. 

<div class="page-break"></div>

# Explainability & Insights

What is the model actually looking at? 

### fNIRS Leads the Way
- **SHAP values** show that the model heavily relies on fNIRS HbO summary features (skew, early-window response, slope, max).
- **Attention Gates** confirm this, with the learned fusion gate leaning slightly more towards fNIRS overall (0.514 vs 0.478 for EEG).

### The Role of EEG
While fNIRS carries the most stable signal, EEG band-power features (alpha, beta, gamma) still provide essential supporting cues, particularly for classifying High load states.

<div class="page-break"></div>

# The Live Demo & Serving

This isn't just a Jupyter notebook. The final locked model is served in a production-ready **FastAPI + React** application.

- **Traceable:** Every metric, confusion matrix, and insight in the UI traces back to a committed artifact generated during the training phase.
- **Live Inference:** The app features an Upload Data page that accepts CSVs of extracted TU Berlin features, preprocessing and predicting in real-time.
- **Visualization:** Playback real held-out test windows to watch the model's probabilities and attention weights shift second by second.
