# 🎯 Quick Start Guide for MDD Biomarker Discovery Project

## 🚀 Getting Started in 5 Minutes

### Step 1: Environment Setup

```powershell
# Navigate to project directory
cd "m:\5th sem\ML2-project"

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 2: Prepare Your Data

```powershell
# Create data directory structure
New-Item -ItemType Directory -Force -Path data\raw\audio
New-Item -ItemType Directory -Force -Path data\raw\transcripts
New-Item -ItemType Directory -Force -Path data\processed
New-Item -ItemType Directory -Force -Path data\features
```

**Place your dataset files:**
- Audio files (`.wav`) → `data/raw/audio/`
- Text transcripts (`.txt`) → `data/raw/transcripts/`
- Metadata CSV → `data/raw/metadata.csv`

### Step 3: Run the Pipeline

```powershell
# Full pipeline execution
python main.py --mode full

# Or run step-by-step
python main.py --mode preprocess
python main.py --mode feature_extraction
python main.py --mode clustering
```

### Step 4: Explore Results

```powershell
# Launch Jupyter notebook
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

---

## 📊 Dataset: DAIC-WOZ

### How to Obtain

1. **Request Access:**
   - Visit: https://dcapswoz.ict.usc.edu/
   - Fill out data request form
   - Wait for approval (usually 1-2 weeks)

2. **Download:**
   - Follow instructions in approval email
   - Download audio, video, and transcript files
   - Download PHQ-8 scores CSV

3. **Organize:**
```
data/raw/
├── audio/
│   ├── 300_P.wav
│   ├── 301_P.wav
│   └── ...
├── transcripts/
│   ├── 300_TRANSCRIPT.txt
│   ├── 301_TRANSCRIPT.txt
│   └── ...
└── metadata.csv (with PHQ-8 scores)
```

### Alternative Datasets

**If you can't access DAIC-WOZ, use:**

1. **Kaggle Depression Dataset:**
   - https://www.kaggle.com/datasets/arashnic/the-depression-dataset
   - Text-only, easier to start with

2. **Create Synthetic Data (for testing):**
   ```python
   python scripts/generate_synthetic_data.py
   ```

---

## 🧪 Example Workflow

### Minimal Working Example

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load your features (replace with actual feature extraction)
features = np.random.randn(100, 50)  # 100 samples, 50 features

# Dimensionality reduction
pca = PCA(n_components=10)
reduced_features = pca.fit_transform(features)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(reduced_features)

print(f"Found {len(set(labels))} clusters")
print(f"Cluster sizes: {np.bincount(labels)}")
```

---

## 📝 Project Checklist

### Week 1: Setup & Research
- [ ] Install all dependencies
- [ ] Request DAIC-WOZ access
- [ ] Read 5-10 key papers on depression ML
- [ ] Understand project goals
- [ ] Create project proposal document

### Week 2: Preprocessing & Features
- [ ] Download and organize dataset
- [ ] Run audio preprocessing
- [ ] Run text preprocessing
- [ ] Extract acoustic features
- [ ] Extract linguistic features
- [ ] Combine multimodal features

### Week 3: Dimensionality Reduction & Clustering
- [ ] Apply PCA
- [ ] Apply t-SNE/UMAP
- [ ] Train VAE (optional)
- [ ] Run K-Means clustering
- [ ] Run GMM clustering
- [ ] Run Spectral clustering
- [ ] Evaluate clustering quality

### Week 4: Analysis & Documentation
- [ ] Analyze biomarker patterns per cluster
- [ ] Statistical validation (ANOVA, etc.)
- [ ] Create all visualizations
- [ ] Write research paper
- [ ] Create presentation
- [ ] Prepare final submission

---

## 🎨 Visualization Examples

### Key Plots to Generate

1. **t-SNE Cluster Visualization:**
```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedded = tsne.fit_transform(features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedded[:, 0], embedded[:, 1], 
                     c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('t-SNE Visualization of Depression Subtypes')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```

2. **Feature Heatmap:**
```python
import seaborn as sns

# Calculate mean features per cluster
cluster_means = pd.DataFrame(features).groupby(labels).mean()

plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means.T, cmap='coolwarm', center=0, 
            xticklabels=[f'Cluster {i}' for i in range(len(cluster_means))],
            cbar_kws={'label': 'Feature Value'})
plt.title('Biomarker Profiles by Cluster')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
```

3. **PHQ Score Distribution:**
```python
plt.figure(figsize=(10, 6))
for i in range(len(set(labels))):
    cluster_scores = phq_scores[labels == i]
    plt.hist(cluster_scores, alpha=0.5, label=f'Cluster {i}', bins=15)

plt.xlabel('PHQ-8 Score')
plt.ylabel('Frequency')
plt.title('Depression Severity Distribution by Cluster')
plt.legend()
plt.show()
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: "No module named 'librosa'"**
```powershell
pip install librosa soundfile
```

**2. NLTK Data Not Found**
```python
import nltk
nltk.download('all')  # Download all NLTK data
```

**3. Memory Error with Large Dataset**
```python
# Process in batches
batch_size = 100
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    # Process batch
```

**4. VAE Not Training**
- Check learning rate (try 1e-4 to 1e-3)
- Ensure features are normalized
- Increase batch size
- Try simpler architecture first

---

## 📚 Recommended Reading

### Must-Read Papers

1. **DAIC-WOZ Dataset:**
   - Gratch et al. (2014) - "The Distress Analysis Interview Corpus"

2. **Depression Detection:**
   - Cummins et al. (2015) - "Review of depression assessment using speech"
   - Alhanai et al. (2018) - "Detecting depression with audio/text"

3. **Clustering in Psychiatry:**
   - Drysdale et al. (2017) - "fMRI-based depression subtypes" (Nature Medicine)

### Online Resources

- Scikit-learn Documentation: https://scikit-learn.org/
- Librosa Tutorial: https://librosa.org/doc/latest/tutorial.html
- VAE Tutorial: https://pytorch.org/tutorials/

---

## 💡 Tips for Success

### Academic Submission

1. **Focus on interpretation:** Don't just report numbers—explain what they mean clinically
2. **Statistical rigor:** Always report p-values, confidence intervals, effect sizes
3. **Reproducibility:** Provide random seeds, detailed parameters
4. **Figures:** High-quality, publication-ready plots (300 DPI minimum)

### Competition/Hackathon

1. **Live demo:** Build a simple web interface with Streamlit
2. **Story telling:** Lead with motivation, end with impact
3. **Code quality:** Clean, documented, modular code
4. **Innovation:** Add unique feature (explainability, real-time processing, etc.)

---

## 🤝 Need Help?

### Resources

- GitHub Issues: Create an issue for bugs/questions
- Stack Overflow: Tag `machine-learning`, `depression-detection`
- Paper References: See `docs/paper/references.bib`

### Contact

- Email: [Your email]
- LinkedIn: [Your profile]

---

## 🎉 Project Complete!

**Congratulations!** You now have a complete, publication-quality project on MDD biomarker discovery.

### Next Steps:

1. ✅ Run full pipeline on real data
2. ✅ Generate all visualizations
3. ✅ Write research paper
4. ✅ Create presentation
5. ✅ Submit to competition/journal/conference

**Good luck with your research!** 🚀
