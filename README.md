# 🩺 Unsupervised Discovery of Hidden Biomarkers and Subtypes for Major Depressive Disorder

> **A multimodal machine learning approach to mental health diagnostics**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Project Overview

This project applies **unsupervised machine learning** techniques (clustering, PCA, autoencoders) on behavioral, speech, and neuroimaging data to identify **hidden subtypes** and **biomarker patterns** associated with Major Depressive Disorder (MDD).

### Why This Matters

Depression is invisible, varied, and often misdiagnosed. Traditional diagnosis relies on subjective questionnaires. This project uses AI to uncover objective, data-driven patterns in:
- 🎤 Voice tremors and speech pauses
- 📝 Linguistic patterns and emotional tone
- 🧠 Neural activity signatures

**Goal:** Move mental health diagnosis from subjective to objective, from generalized to personalized.

---

## 🎯 Research Questions

1. Can unsupervised algorithms detect **meaningful latent subtypes** of MDD patients?
2. What **biomarkers** (speech, text, EEG/fMRI features) define these subtypes?
3. Do discovered clusters correlate with **depression severity** or symptom patterns?
4. Can dimensionality reduction capture **hidden emotional representations**?

---

## 📊 Dataset Options

### Primary: DAIC-WOZ Dataset (Recommended)
- **Contains:** Audio, facial expressions, text transcripts, PHQ-8 scores
- **Best for:** Speech + emotion biomarker detection
- **Source:** USC Institute for Creative Technologies

### Alternative Options:
- **OpenNeuro ds002748:** fMRI scans (brain biomarkers)
- **Kaggle Depression Survey:** Text + questionnaires (NLP-focused)

---

## 🏗️ Project Architecture

```
Raw Multimodal Data
    ↓
Preprocessing Pipeline
    ↓
Feature Engineering
    ↓
Dimensionality Reduction (PCA/VAE/t-SNE)
    ↓
Clustering (K-Means/GMM/Spectral)
    ↓
Biomarker Analysis & Interpretation
    ↓
Visualization & Research Paper
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd "m:\5th sem\ML2-project"

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

```bash
# Place your dataset in the data/ folder
data/
  ├── raw/
  │   ├── audio/
  │   ├── transcripts/
  │   └── metadata.csv
  └── processed/
```

### 3. Run the Pipeline

```bash
# Full pipeline execution
python main.py --dataset daic-woz --mode full

# Or run individual steps
python main.py --mode preprocess
python main.py --mode feature_extraction
python main.py --mode clustering
```

### 4. Explore Results

```bash
# Launch Jupyter notebook for analysis
jupyter notebook notebooks/01_exploratory_analysis.ipynb

# Generate visualizations
python scripts/generate_visualizations.py
```

---

## 📁 Project Structure

```
ML2-project/
│
├── data/                          # Data directory
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Preprocessed data
│   └── features/                  # Extracted features
│
├── src/                           # Source code
│   ├── preprocessing/             # Data preprocessing modules
│   │   ├── audio_processor.py
│   │   ├── text_processor.py
│   │   └── neuroimaging_processor.py
│   │
│   ├── features/                  # Feature extraction
│   │   ├── audio_features.py
│   │   ├── text_features.py
│   │   └── multimodal_fusion.py
│   │
│   ├── models/                    # ML models
│   │   ├── dimensionality_reduction.py
│   │   ├── clustering.py
│   │   └── autoencoder.py
│   │
│   ├── analysis/                  # Analysis tools
│   │   ├── biomarker_analysis.py
│   │   └── cluster_interpretation.py
│   │
│   └── visualization/             # Visualization utilities
│       ├── plots.py
│       └── dashboard.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_dimensionality_reduction.ipynb
│   └── 04_clustering_analysis.ipynb
│
├── scripts/                       # Utility scripts
│   ├── download_data.py
│   ├── train_models.py
│   └── generate_visualizations.py
│
├── docs/                          # Documentation
│   ├── paper/                     # Research paper
│   │   ├── main.tex
│   │   └── references.bib
│   └── presentation/              # Slides
│       └── presentation.pptx
│
├── results/                       # Output results
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── tests/                         # Unit tests
│   └── test_preprocessing.py
│
├── main.py                        # Main pipeline script
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🧰 Technologies Used

### Core ML/AI
- **Scikit-Learn** - Clustering, PCA, preprocessing
- **PyTorch** - Autoencoder/VAE implementation
- **TensorFlow** - Alternative deep learning framework

### Signal Processing
- **Librosa** - Audio feature extraction
- **MNE** - EEG/MEG analysis
- **Nilearn** - fMRI processing

### NLP
- **HuggingFace Transformers** - BERT embeddings
- **NLTK** - Text preprocessing
- **spaCy** - Advanced NLP

### Visualization
- **Matplotlib/Seaborn** - Static plots
- **Plotly** - Interactive visualizations
- **Yellowbrick** - ML visualization

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **SciPy** - Scientific computing

---

## 🔬 Methodology

### 1. Data Preprocessing
- Audio: Convert to mel-spectrograms, extract MFCC features
- Text: Clean, tokenize, generate embeddings (TF-IDF/BERT)
- fMRI: Extract ROI time-series, compute connectivity matrices

### 2. Feature Engineering
- Standardization with `StandardScaler`
- Multimodal feature fusion
- Outlier removal using Isolation Forest

### 3. Dimensionality Reduction
- **PCA**: Linear variance-based reduction
- **t-SNE/UMAP**: Nonlinear manifold visualization
- **VAE**: Deep learning-based latent representations

### 4. Clustering Algorithms
- **K-Means**: Baseline clustering
- **Gaussian Mixture Models**: Soft clustering for fuzzy states
- **Spectral Clustering**: Graph-based clustering for complex patterns

### 5. Biomarker Analysis
- Cluster characterization by feature means
- Correlation with PHQ-8/9 scores
- Statistical significance testing

---

## 📈 Expected Outcomes

### Discoveries
- ✅ 2-4 hidden subtypes of depression
- ✅ Biomarkers defining each subtype
- ✅ Correlation between features and severity
- ✅ Evidence for ML-based diagnosis

### Deliverables
- 📄 Research paper (6-10 pages)
- 📊 Presentation (8-10 slides)
- 💻 Jupyter notebooks with experiments
- 📉 Comprehensive visualizations
- 📋 Cluster interpretation report

---

## 🗓️ Project Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| **Week 1** | Research + Dataset Preparation | Background study, data download |
| **Week 2** | Preprocessing + Feature Extraction | Clean dataset, feature matrices |
| **Week 3** | Dimensionality Reduction + Clustering | Results, cluster assignments |
| **Week 4** | Analysis + Documentation | Paper, presentation, final report |

---

## 🎨 Advanced Features (Optional)

- 🌐 **Web Dashboard**: Interactive cluster explorer
- 🔍 **Explainable AI**: SHAP values for biomarker importance
- 🎵 **Audio Spectrograms**: Emotion visualization
- 🔄 **VAE Interpolation**: Smooth transitions between emotional states
- 📱 **Mobile App**: Depression screening tool prototype

---

## 📚 References

1. Gratch, J., et al. (2014). *The Distress Analysis Interview Corpus of human and computer interviews*. LREC.
2. Cummins, N., et al. (2015). *A review of depression and suicide risk assessment using speech analysis*. Speech Communication.
3. Drysdale, A.T., et al. (2017). *Resting-state connectivity biomarkers define neurophysiological subtypes of depression*. Nature Medicine.

---

## 👥 Contributors

**Paramjit** - Lead Researcher & Developer

---

## 📄 License

MIT License - Feel free to use this for research, education, or competition purposes.

---

## 🙏 Acknowledgments

- USC Institute for Creative Technologies (DAIC-WOZ dataset)
- OpenNeuro community
- Mental health research community

---

## 📞 Contact

For questions, collaboration, or support:
- 📧 Email: [Your email]
- 🔗 LinkedIn: [Your profile]
- 💻 GitHub: [Your username]

---

**⚠️ Ethical Note:** This project is for research purposes only. It is not intended to replace professional medical diagnosis or treatment. If you or someone you know is experiencing depression, please seek help from qualified mental health professionals.

**Crisis Resources:**
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741

---

*"In the silence of data, we find the voice of invisible pain."*
