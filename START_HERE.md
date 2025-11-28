# 🚀 Quick Start: Your Mental Health Dataset Analysis

## ✅ You're Ready to Start!

Your dataset is loaded and ready: `data/Combined Data.csv`

**Dataset:** Mental Health Sentiment Analysis (94,025 text statements)  
**Source:** Kaggle  
**Labels:** Anxiety, Depression, Normal, Bipolar, Personality Disorder, Stress, Suicidal

---

## 🎯 Run the Analysis NOW

### Option 1: Jupyter Notebook (Recommended)

```powershell
# Open the custom notebook made for your dataset
jupyter notebook notebooks/02_mental_health_dataset_analysis.ipynb
```

Then **run all cells** (Cell → Run All) and watch the magic happen! ✨

### Option 2: Step-by-Step

Run each cell in the notebook to:
1. ✅ Load your 94K+ mental health statements
2. ✅ Extract linguistic biomarkers (word counts, emotional words, etc.)
3. ✅ Create TF-IDF features
4. ✅ Apply PCA dimensionality reduction
5. ✅ Visualize with t-SNE
6. ✅ Discover hidden clusters with K-Means
7. ✅ Analyze biomarker patterns per cluster
8. ✅ Validate correlation with mental health labels

---

## 📊 What You'll Discover

### Clustering Results
- **Optimal number of clusters** (found automatically)
- **Cluster quality metrics** (Silhouette, Davies-Bouldin)
- **Statistical validation** (Chi-square test)

### Biomarker Patterns
- Negative word ratio per cluster
- First-person pronoun usage
- Average statement length
- Emotional word frequency

### Visualizations
- 📈 t-SNE scatter plots showing hidden patterns
- 🔥 Heatmaps of biomarker profiles
- 📊 Cluster distribution by mental health status
- 📉 Elbow curves for optimal k

---

## 🎓 For Your Project Report

The notebook generates everything you need:

### Results Section
- Cluster statistics (automatically calculated)
- Evaluation metrics (Silhouette, etc.)
- Statistical significance (p-values)

### Figures
- All plots are publication-ready
- Save them with: `plt.savefig('figure.png', dpi=300)`

### Tables
- Feature comparison tables
- Cluster distribution tables
- Confusion matrices

---

## 💡 Tips

### Fast Testing
- Notebook uses 5,000 samples for t-SNE (faster computation)
- Uses full 94K dataset for clustering

### Customization
- Change `best_k` to try different cluster numbers
- Modify `max_features` in TF-IDF for more/fewer features
- Adjust `perplexity` in t-SNE for different visualizations

### Save Results
```python
# Save cluster labels
df[['statement', 'status', 'cluster']].to_csv('results/clustered_data.csv')

# Save figures
plt.savefig('results/figures/tsne_clusters.png', dpi=300, bbox_inches='tight')
```

---

## 📝 Expected Results

Based on your dataset structure, you should discover:

1. **Multiple clusters** within each mental health condition
2. **Linguistic differences** between anxiety, depression, stress, etc.
3. **Significant correlation** between clusters and diagnoses (p < 0.05)
4. **Biomarker patterns** like:
   - Anxiety: High negative word ratio, present-tense focus
   - Depression: High first-person pronouns, past-tense focus
   - Suicidal: Specific trigger words, hopelessness indicators

---

## 🐛 Troubleshooting

### "Module not found"
```powershell
pip install -r requirements.txt
```

### "NLTK data not found"
The notebook downloads it automatically on first run.

### "Memory error"
Reduce sample size in t-SNE section:
```python
sample_size = 1000  # Instead of 5000
```

---

## 🎯 Next Steps After Running

1. ✅ **Analyze** cluster characteristics
2. ✅ **Save** all figures for your report
3. ✅ **Write** findings in research paper template
4. ✅ **Create** presentation slides
5. ✅ **Try** other algorithms (GMM, Spectral)

---

## 🏆 Your Project Checklist

- [x] Dataset downloaded ✓
- [x] Notebook created ✓
- [ ] Run full analysis
- [ ] Interpret clusters
- [ ] Create visualizations
- [ ] Write paper
- [ ] Prepare presentation

---

**Ready? Open the notebook and start discovering hidden patterns in mental health!** 🧠✨

```powershell
jupyter notebook notebooks/02_mental_health_dataset_analysis.ipynb
```
