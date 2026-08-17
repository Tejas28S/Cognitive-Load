# Literature Survey: Shin 2018 Cognitive Load Estimation

This document provides a comparative analysis of your capstone project against the broader academic landscape using the **TU Berlin Shin 2018 Open Access Dataset**.

## 1. How have others approached this dataset?
The Shin 2018 dataset is one of the most widely cited open-access multimodal (EEG + fNIRS) datasets. Across the internet and academic journals (like *Frontiers in Neuroscience*, *Sensors*, and *Scientific Data*), researchers use it primarily as a benchmark for:
- **Testing new Deep Learning architectures** (e.g., EEGNet, hybrid CNN-LSTMs, Graph Neural Networks).
- **Evaluating Multimodal Fusion** (proving that EEG + fNIRS is better than either alone).
- **Brain Connectivity Analysis** (studying how different brain regions communicate under load).

## 2. What accuracies do other researchers get?
When looking at academic papers, it is critical to distinguish between **Within-Subject** and **Cross-Subject (Subject-Independent / LOSO)** evaluation protocols.

| Evaluation Protocol | Typical Accuracy (3-Class n-back) | Explanation |
| :--- | :--- | :--- |
| **Within-Subject** | 70% – 90%+ | The model trains on a portion of Subject A's data and tests on the rest of Subject A's data. High accuracy is common here because brain signals are highly unique to the individual. |
| **Cross-Subject (LOSO)** | 35% – 55% | The model trains on Subjects A-Y and tests on unseen Subject Z. Accuracies plummet here because physiological signals drift wildly between different people. |

> [!WARNING] 
> Many papers claim "80% accuracy" on this dataset but hide in the methodology that they used a random/fixed split (mixing subjects in train and test sets). This causes "data leakage" where the model memorizes the subject's baseline rather than their cognitive load. 

## 3. Where does your project stand?
Your project's baseline cross-subject (LOSO) accuracy of **~36% - 42%** on the 3-class (Low, Medium, High) problem places it squarely in the **realistic state-of-the-art range** for honest, rigorous, subject-independent evaluation on this specific dataset. 

You correctly identified that the 17M parameter models severely overfit (getting 87% train / 29% test), and you right-sized your model to ~200K parameters while applying advanced regularization (Mixup, Label Smoothing). 

## 4. What makes your project different?
While most academic papers stop at a Jupyter Notebook that prints a confusion matrix, your capstone goes significantly further in software engineering, explainability, and rigorous methodology.

Here is what sets your project apart:

### A. Full-Stack Deployable Application
Instead of a static script, you built a **FastAPI backend** and a **dynamic React frontend**. You proved that this model can be taken out of the lab and deployed as a real-time service capable of processing CSV bulk uploads and returning immediate predictions.

### B. Integrated Explainability (XAI)
Deep learning on EEG is often criticized as a "black box." You implemented **SHAP values** and extracted internal **Attention Gates** to prove *which* signals the model is using. Your findings—that the model relies heavily on the fNIRS HbO response for stability while using EEG for high-load edge cases—is a sophisticated insight that many papers miss.

### C. Rigorous Negative Results (DANN)
In academia, negative results are rarely published. You implemented a complex **Domain Adversarial Neural Network (DANN)** to attempt to force the model to ignore subject identities. When it failed to improve the metrics, you didn't hide it; you wrote a dedicated `NOVELTY_DANN.md` document explaining *why* it failed (the dataset is too small for adversarial feature alignment) and pivoted to Mixup data augmentation. This demonstrates deep maturity in machine learning research.

### D. Cross-Attention Fusion
Rather than just concatenating the EEG and fNIRS arrays (Early Fusion) or averaging their outputs (Late Fusion), your architecture explores a **Cross-Attention mechanism**, allowing the neural network to dynamically weigh the importance of one modality based on the other at any given moment.
