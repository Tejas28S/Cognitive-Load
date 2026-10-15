# Frontend Architecture & Presentation Guide

This document outlines the architectural stack of the Cognitive Load web application and provides a structured guide on what information to present during the Capstone project presentation.

---

## 1. Tools & Architecture Overview

The frontend is built as a Single Page Application (SPA) prioritizing fast performance, data-rich visualizations, and a modern aesthetic. 

### Core Tech Stack
* **React 19 & TypeScript**: The foundation of the user interface. React provides a robust component-driven architecture, while TypeScript ensures type safety, reducing runtime errors when handling complex scientific data (e.g., multi-channel EEG/fNIRS signals).
* **Vite 8**: Serves as the build tool and development server, offering instantaneous Hot Module Replacement (HMR) and optimized, minified production builds.
* **Tailwind CSS v4**: Utility-first CSS framework used for all styling. It enables a highly responsive design, custom design tokens, and a seamless Light/Dark mode toggle without writing custom CSS files for every component.
* **React Router v7**: Handles client-side routing. It enables smooth navigation across the application's distinct sections (Assess, Evidence, About) without page reloads.

### Data & Interactivity
* **Recharts**: The primary library for rendering complex data visualizations. It is used extensively to display real-time signal traces, confusion matrices, class recall charts, and SHAP value distributions.
* **Framer Motion**: Powers the application's micro-interactions, page transitions, and reveal animations, contributing to a premium, "living" interface.
* **Axios**: The HTTP client responsible for communicating with the Python backend. It handles fetching model predictions, run history, and evaluation metrics.
* **PDF Export (`jspdf` & `html-to-image`)**: Allows users to snapshot DOM elements (like charts or reports) and export them as downloadable PDF documents.

### Quality Assurance
* **Playwright**: Used for End-to-End (E2E) testing, specifically smoke testing critical user flows like rendering the dashboard and navigating routes.
* **Oxlint**: An extremely fast linter used to enforce code quality and standard practices across the codebase.

---

## 2. Application Structure

The application is structurally divided to support different user intents:
1. **The Executive View (Dashboard)**: Answers "how good is the model?" immediately.
2. **Assess / Inference**: `Live Prediction`, `Session Trend`, and `Upload Data`. Allows users to interact with the model directly.
3. **Evidence**: `Model Insights`, `Evaluation Metrics`, and `Run Archive`. Provides the deep quantitative record and explainability (SHAP, attention).
4. **Context**: `Overview`, `Workflow`, `Methodology`, and `About`. Explains the science and the architecture.

---

## 3. Presentation Guidelines: What to Show

During your Capstone presentation, the frontend demonstration should focus on **usability, data visualization, and model explainability**. 

Follow this structure for the live demo portion:

### A. The Hook: Landing & Dashboard (2 mins)
* **What to show**: Start on the Landing page, then transition to the Dashboard.
* **What to say**: Explain that the application is designed to make complex cognitive load models accessible. Highlight the top-level KPIs (Accuracy, F1-Score) on the Dashboard. Emphasize that the dashboard immediately proves the model's validity.

### B. Architecture & Tech Stack summary (1 min)
* **What to show**: `Command Palette (⌘K)` or navigating the top navbar.
* **What to say**: Briefly mention that the frontend is a React/Vite SPA styled with Tailwind. Explain that it communicates with the FastAPI backend in real-time, fetching live inference data and historical run metrics. 

### C. Live Inference & Visualization (3 mins)
* **What to show**: Navigate to **Live Prediction** or **Upload Data**.
* **What to say**: 
  * Demonstrate how a user submits a window of EEG/fNIRS data to the model.
  * Point out the **Recharts** visualizations showing the signal traces. 
  * Highlight the speed of the interface (thanks to React/Vite) and the real-time feedback.

### D. Model Explainability & Trust (2 mins)
* **What to show**: Navigate to **Model Insights** (Evidence section).
* **What to say**: 
  * Show the Confusion Matrix and Class Recall charts.
  * Show the SHAP value distributions.
  * **Crucial Point**: Explain that a healthcare/cognitive model *must* be explainable. Point out how the frontend makes SHAP values easy to interpret for non-technical stakeholders.

### E. Conclusion & Export (1 min)
* **What to show**: Demonstrate the PDF export functionality (if demonstrating a report generation).
* **What to say**: Conclude by showing how researchers or clinicians can export these visualizations into a concrete report, bridging the gap between a machine learning model and real-world utility.
