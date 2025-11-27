# 📊 Presentation Outline: MDD Biomarker Discovery

## Slide-by-Slide Guide (8-10 Slides)

---

### **Slide 1: Title Slide**

**Title:** Unsupervised Discovery of Hidden Biomarkers and Subtypes for Major Depressive Disorder using Multimodal Machine Learning

**Subtitle:** Using AI to Reveal the Invisible Patterns of Depression

**Author:** [Your Name]  
**Institution:** [Your University]  
**Date:** November 2025

**Visual:** Abstract neural network or brain visualization with data flowing through it

---

### **Slide 2: The Problem**

**Title:** Depression: The Silent Epidemic

**Content:**
- 280+ million people affected worldwide 🌍
- Current diagnosis is **subjective** (questionnaires, interviews)
- **Heterogeneous** condition - same symptoms, different causes
- Treatment is often **one-size-fits-all**

**Visual:** Infographic showing depression statistics + image of traditional diagnosis (checkbox questionnaire)

**Talking Points:**
- "Imagine diagnosing diabetes without blood sugar tests"
- "That's where mental health is today"

---

### **Slide 3: Our Solution**

**Title:** From Subjective to Objective: AI-Powered Biomarkers

**Content:**
- **Objective** measurement through multimodal data
- **Personalized** subtypes based on hidden patterns
- **Data-driven** approach using unsupervised ML

**Three Pillars:**
1. 🎤 **Speech Analysis** - Acoustic features
2. 📝 **Language Patterns** - Linguistic markers  
3. 🧠 **Machine Learning** - Subtype discovery

**Visual:** Three-column diagram showing data → AI → insights

---

### **Slide 4: Dataset & Methodology**

**Title:** Data Pipeline Architecture

**Dataset:**
- **DAIC-WOZ:** 189 clinical interviews
- **Modalities:** Audio + Text transcripts
- **Labels:** PHQ-8 depression scores

**Pipeline:**
```
Raw Data → Preprocessing → Feature Extraction →
Dimensionality Reduction → Clustering → Analysis
```

**Visual:** Flowchart with icons for each stage

**Key Numbers:**
- 131 multimodal features extracted
- 3 dimensionality reduction methods (PCA, t-SNE, VAE)
- 3 clustering algorithms tested

---

### **Slide 5: Feature Extraction**

**Title:** What We Measured: 131 Biomarkers

**Two Columns:**

**Left - Acoustic Features (89):**
- Pitch (mean, std, range)
- Energy & tempo
- Speech pauses
- MFCC coefficients
- Spectral features

**Right - Linguistic Features (42):**
- Word count & complexity
- Emotional words (negative/positive)
- First-person pronoun usage
- Sentiment polarity
- Filler words & pauses

**Visual:** Feature extraction process with audio waveform → features, text → features

---

### **Slide 6: Discovering Subtypes**

**Title:** Unsupervised Learning Reveals [X] Hidden Subtypes

**Main Visual:** t-SNE/UMAP scatter plot showing distinct clusters, color-coded

**Key Results:**
- **Best Model:** [K-Means/GMM] with **[X] clusters**
- **Silhouette Score:** 0.XX
- **Clinical Validation:** Significant correlation with PHQ-8 (p < 0.001)

**Visual:** Include both scatter plot AND small comparison table of clustering methods

---

### **Slide 7: Subtype Profiles**

**Title:** The [X] Faces of Depression

**Layout:** [X] boxes, one per cluster

**For Each Subtype (example for 3 clusters):**

**Cluster 1: "Cognitive Subtype" (30%)**
- 🔸 Low pitch variability
- 🔸 High pause ratio
- 🔸 Elevated first-person pronouns
- **PHQ Score:** μ = 18.3

**Cluster 2: "Emotional Exhaustion" (45%)**
- 🔸 Low speech energy
- 🔸 High negative words
- 🔸 Reduced speech tempo
- **PHQ Score:** μ = 21.7

**Cluster 3: "Mild Symptoms" (25%)**
- 🔸 Normal prosody
- 🔸 Moderate speech rate
- 🔸 Lower negative ratio
- **PHQ Score:** μ = 12.1

**Visual:** Radar chart comparing 3 clusters on key biomarkers

---

### **Slide 8: Key Biomarkers**

**Title:** Top 10 Discriminative Features

**Visual:** Horizontal bar chart showing feature importance

**Feature Importance (example):**
1. Pause ratio - 0.87
2. Pitch mean - 0.82
3. Negative word ratio - 0.79
4. First-person pronouns - 0.76
5. Speech energy - 0.73
6. ...

**Insight Box:**
"These patterns are **consistent across subtypes** but **differ in degree**"

---

### **Slide 9: Clinical Validation**

**Title:** Statistical Validation & Correlation

**Two Visuals Side-by-Side:**

**Left:** Box plot showing PHQ-8 score distribution by cluster
- ANOVA: F = XX.X, p < 0.001
- Significant differences between all cluster pairs

**Right:** Confusion matrix or correlation heatmap
- Clusters vs. Depression Severity Categories (None/Mild/Moderate/Severe)

**Key Finding:**
"Unsupervised clusters align with clinical severity, validating their diagnostic relevance"

---

### **Slide 10: Impact & Future Work**

**Title:** Transforming Mental Health Diagnosis

**Impact:**
✅ **Objective** biomarkers for depression screening  
✅ **Personalized** treatment based on subtype  
✅ **Early detection** through continuous monitoring  
✅ **Reduced stigma** via scientific measurement

**Future Directions:**
1. 🔬 Validate on larger, diverse datasets
2. 🧠 Incorporate neuroimaging (fMRI, EEG)
3. 📱 Develop mobile screening app
4. 🏥 Clinical trials for subtype-specific treatments

**Visual:** Impact infographic with icons

---

### **Slide 11: Conclusion (Final Slide)**

**Title:** From Invisible to Measurable

**Key Takeaways:**
1. ML can discover **hidden depression subtypes** from behavioral data
2. **Objective biomarkers** exist in speech and language  
3. **Data-driven personalization** is possible in mental health

**Closing Statement:**
*"Depression has lived in shadows for too long. Today, we bring it into the light of data."*

**Contact Information:**
- Email: [your.email@university.edu]
- GitHub: github.com/[username]/mdd-biomarkers
- Paper: [arXiv/journal link]

**Visual:** Hopeful image - brain with light/neural network transforming into clarity

---

## 🎤 Presentation Tips

### Delivery (10-15 minutes)

**Timing:**
- Slides 1-3: 2 minutes (setup problem)
- Slides 4-5: 3 minutes (methodology)
- Slides 6-8: 5 minutes (results - spend most time here!)
- Slides 9-10: 3 minutes (validation & impact)
- Slide 11: 1 minute (conclusion)
- Q&A: 5 minutes

### What to Emphasize

1. **Problem first:** Make audience care about the problem
2. **Visual results:** Let the plots tell the story
3. **Clinical relevance:** Always tie back to real patients
4. **Simplicity:** Avoid jargon, explain technical terms

### Common Questions to Prepare For

1. "How did you validate clinical relevance?"
   - PHQ-8 correlation, statistical tests, expert interpretation

2. "Why unsupervised instead of supervised learning?"
   - Discover hidden patterns, no label bias, more generalizable

3. "What about sample size?"
   - 189 patients is standard for this dataset, plan to validate on larger cohorts

4. "Can this replace clinicians?"
   - No! It's a **screening tool** and **decision support**, not replacement

5. "What about ethical considerations?"
   - De-identified data, consent, transparent AI, clinical oversight required

---

## 🎨 Design Guidelines

### Color Palette
- **Primary:** Deep blue (#2C3E50) - trust, clinical
- **Accent:** Teal (#16A085) - innovation, hope
- **Highlights:** Orange (#E67E22) - energy, important points
- **Background:** White or light gray

### Fonts
- **Headers:** Montserrat Bold or Roboto Bold
- **Body:** Open Sans or Roboto Regular
- **Code:** Consolas or Monaco

### Visual Style
- **Consistent:** Same style across all slides
- **Clean:** White space is your friend
- **Professional:** Academic but engaging
- **Data-driven:** Let visualizations dominate

---

## 📁 Presentation Files to Prepare

1. **PowerPoint/Keynote:** Main presentation
2. **PDF version:** For distribution
3. **Backup slides:** Extra technical details (in appendix)
4. **Live demo (optional):** Interactive visualization
5. **Handout:** One-page summary with key findings

---

**Good luck with your presentation!** 🎯🚀
