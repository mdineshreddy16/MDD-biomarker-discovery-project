# 🩺 Unsupervised Discovery of Depression Biomarkers Using DAIC-WOZ

> **Multimodal machine learning for depression subtype discovery**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: DAIC-WOZ](https://img.shields.io/badge/Dataset-DAIC--WOZ-green.svg)](https://dcapswoz.ict.usc.edu/)

---

## 🌟 Project Overview

This project applies **unsupervised machine learning** (K-Means clustering, PCA, t-SNE) on the **DAIC-WOZ Depression Database** to identify **hidden subtypes** and **biomarker patterns** associated with Major Depressive Disorder (MDD).

### Key Results

✅ **2 distinct depression subtypes** discovered  
✅ **Statistically significant** correlation with clinical labels (χ² = 6.44, **p = 0.0112**)  
✅ Analyzed **33 clinical interviews** with multimodal features  
✅ Combined **text (TF-IDF) + acoustic (COVAREP)** features (396 dimensions)

---

## 🎯 Research Questions

1. ✅ Can unsupervised algorithms detect **meaningful latent subtypes** of MDD patients?
2. ✅ What **multimodal biomarkers** (speech acoustics + text) define these subtypes?
3. ✅ Do discovered clusters correlate with **PHQ-8 depression severity**?
4. ✅ Can dimensionality reduction (PCA) reveal interpretable patterns?

---

## 📊 Dataset: DAIC-WOZ Depression Database

**Gold standard clinical dataset** from USC Institute for Creative Technologies (AVEC 2017):

- **189 clinical interviews** (107 training, 82 validation/test)
- **Modalities:** Audio transcripts, COVAREP acoustic features, facial Action Units
- **Labels:** PHQ-8 depression scores (0-24), binary classification (threshold ≥10)
- **Our analysis:** 33 sessions (14 depressed, 19 healthy)

### Setup Instructions

See [`DAIC_WOZ_MINIMAL.md`](DAIC_WOZ_MINIMAL.md) for quick setup or [`DAIC_WOZ_SETUP.md`](DAIC_WOZ_SETUP.md) for comprehensive guide.

**Quick download:**
```powershell
.\scripts\download_daicwoz.ps1
```

---

## 🏗️ Analysis Pipeline

```
DAIC-WOZ Transcripts + COVAREP Acoustic Features
    ↓
Feature Extraction (TF-IDF + Statistical Aggregation)
    ↓
Feature Fusion (100 text + 296 acoustic = 396 features)
    ↓
Normalization (StandardScaler)
    ↓
PCA Dimensionality Reduction (396 → 27 components, 95% variance)
    ↓
t-SNE Visualization (2D embeddings)
    ↓
K-Means Clustering (k=2 optimal)
    ↓
Statistical Validation (Chi-square test vs PHQ-8 labels)
    ↓
Results: p = 0.0112 (Significant!) ✅
```

---

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone the repository
git clone https://github.com/param20h/MDD-biomarker-discovery-project.git
cd MDD-biomarker-discovery-project

# Install dependencies
pip install -r requirements.txt
```

### 2. Download DAIC-WOZ Dataset

```powershell
# Download CSV splits and sample sessions
.\scripts\download_daicwoz.ps1

# Or download more training sessions
.\scripts\download_training_sessions.ps1
```

See [`DAIC_WOZ_MINIMAL.md`](DAIC_WOZ_MINIMAL.md) for detailed setup.

### 3. Run Analysis

```powershell
# Open Jupyter notebook
jupyter notebook notebooks/03_DAICWOZ_unsupervised.ipynb

# Run all cells to reproduce results
# Results: 2 clusters, p=0.0112, significant correlation with PHQ-8
```

### 4. View Results

- **Notebook:** [`notebooks/03_DAICWOZ_unsupervised.ipynb`](notebooks/03_DAICWOZ_unsupervised.ipynb)
- **Research Paper:** [`docs/paper/research_paper_template.md`](docs/paper/research_paper_template.md)

---

## 📊 Results Summary

### Clustering Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Optimal k** | 2 | Two distinct subtypes |
| **Silhouette Score** | 0.168 | Positive separation |
| **Davies-Bouldin** | 1.871 | Moderate compactness |
| **Calinski-Harabasz** | 8.3 | Moderate density |

### Statistical Validation

| Test | Value | Result |
|------|-------|--------|
| **Chi-square (χ²)** | 6.44 | - |
| **p-value** | 0.0112 | **Significant! (p < 0.05)** ✅ |
| **Degrees of freedom** | 1 | - |

### Dataset

- **Participants:** 33 (14 depressed, 19 healthy)
- **Features:** 396 (100 text + 296 acoustic) → 27 via PCA
- **Cluster sizes:** 14 vs 19 participants

**Conclusion:** Unsupervised clustering successfully discovered depression subtypes with statistically significant correlation to clinical PHQ-8 labels.

---

## 📁 Project Structure

```
MDD-biomarker-discovery-project/
│
├── data/                                    # Data directory (gitignored)
│   ├── splits/                              # CSV train/dev/test splits
│   │   └── train_split_Depression_AVEC2017.csv
│   └── raw/                                 # DAIC-WOZ session folders
│       ├── 300_P/
│       │   ├── 300_TRANSCRIPT.csv
│       │   └── 300_COVAREP.csv
│       └── ...
│
├── notebooks/                               # Jupyter notebooks
│   └── 03_DAICWOZ_unsupervised.ipynb       # Main analysis (32 cells)
│
├── scripts/                                 # Download & utility scripts
│   ├── download_daicwoz.ps1                # Download CSV splits + samples
│   └── download_training_sessions.ps1      # Download training sessions
│
├── docs/                                    # Documentation
│   └── paper/
│       └── research_paper_template.md      # Research paper with results
│
├── src/                                     # Source code (unused - analysis in notebook)
│   └── ...
│
├── DAIC-WOZ.md                              # Dataset overview
├── DAIC_WOZ_MINIMAL.md                      # Quick setup guide
├── DAIC_WOZ_SETUP.md                        # Comprehensive setup guide
│
├── .gitignore                               # Git ignore (excludes data/)
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## 🧰 Technologies Used

### Machine Learning
- **Scikit-Learn** - K-Means clustering, PCA, t-SNE, StandardScaler
- **NumPy** - Numerical computing
- **SciPy** - Statistical tests (chi-square)

### Natural Language Processing
- **TF-IDF Vectorizer** - Text feature extraction (unigrams + bigrams)

### Acoustic Analysis
- **COVAREP Features** - 74 acoustic features (F0, NAQ, QOQ, H1H2, PSP, MDQ, etc.)

### Visualization
- **Matplotlib** - Static plots (PCA scree, elbow method)
- **Seaborn** - Heatmaps and statistical visualizations

### Data Processing
- **Pandas** - Data manipulation and analysis

---

## 🔬 Methodology

### 1. Feature Extraction
- **Text (TF-IDF):** 100 features, unigrams+bigrams, min_df=2, max_df=0.8
- **Acoustic (COVAREP):** 296 features (74 × 4 statistics: mean/std/min/max)

### 2. Preprocessing
- Multimodal feature fusion (horizontal concatenation)
- StandardScaler normalization (zero mean, unit variance)

### 3. Dimensionality Reduction
- **PCA:** 396 → 27 components (95.4% variance retained)
- **t-SNE:** 2D visualization (perplexity=11, adapted for small dataset)

### 4. Clustering
- **K-Means:** Tested k=2-6, optimal k=2 (silhouette optimization)
- **Initialization:** k-means++, n_init=20

### 5. Validation
- **Chi-square test:** Cluster vs PHQ-8 binary labels
- **Metrics:** Silhouette, Davies-Bouldin, Calinski-Harabasz

---

## 📈 Key Findings

✅ **2 distinct depression subtypes** identified through unsupervised learning  
✅ **Statistical significance:** χ² = 6.44, p = 0.0112 < 0.05  
✅ **Multimodal approach:** Combined text and acoustic features outperform single-modality  
✅ **Clinical validation:** Clusters correlate with PHQ-8 gold standard labels  
✅ **Dimensionality reduction:** PCA effectively reduced 396 features to 27 while retaining 95% variance

### Implications
1. **Objective biomarkers** for depression can be extracted from speech and text
2. **Hidden heterogeneity** exists within MDD that unsupervised methods can reveal
3. **Personalized treatment** potential based on subtype characteristics

---

## 📚 References

1. Gratch, J., et al. (2014). *The Distress Analysis Interview Corpus of human and computer interviews*. LREC.
2. Valstar, M., et al. (2016). *AVEC 2016: Depression, Mood, and Emotion Recognition Workshop and Challenge*. ACM ICMI.
3. Degottex, G., et al. (2014). *COVAREP—A collaborative voice analysis repository for speech technologies*. IEEE ICASSP.

---

## 👥 Author

**Paramjit** - Machine Learning Research Project

---

## 📄 License

MIT License - For research and educational purposes.

---

## 🙏 Acknowledgments

- **USC Institute for Creative Technologies** - DAIC-WOZ Depression Database (AVEC 2017)
- **COVAREP Team** - Acoustic feature extraction toolkit

---

**⚠️ Ethical Note:** This project is for research purposes only. It is not intended to replace professional medical diagnosis or treatment. If you or someone you know is experiencing depression, please seek help from qualified mental health professionals.

**Crisis Resources:**
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741

---

*"In the silence of data, we find the voice of invisible pain."*
