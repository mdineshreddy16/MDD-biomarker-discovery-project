# 🩺 Unsupervised Discovery of Hidden Biomarkers for Major Depressive Disorder

## Research Paper Structure

---

## Abstract (150-200 words)

Depression affects millions worldwide, yet diagnosis remains largely subjective, relying on self-reported symptoms and clinical questionnaires. This study applies unsupervised machine learning to discover objective biomarkers and latent subtypes of Major Depressive Disorder (MDD) from multimodal data including speech acoustics, linguistic patterns, and behavioral features. We extracted comprehensive features from audio recordings and text transcripts, applied dimensionality reduction techniques (PCA, t-SNE, VAE), and employed multiple clustering algorithms (K-Means, Gaussian Mixture Models, Spectral Clustering) to identify distinct depression subtypes.

Our analysis revealed **[X]** distinct subtypes characterized by unique biomarker patterns: [describe key findings]. Cluster analysis showed significant correlations with PHQ-8 depression severity scores (p < 0.05) and identified novel acoustic-linguistic biomarker combinations. Key biomarkers included [list 2-3 main biomarkers]. These findings demonstrate that machine learning can uncover objective, data-driven patterns in depression that extend beyond traditional diagnostic criteria, potentially enabling more personalized and effective treatment strategies.

**Keywords:** Major Depressive Disorder, Unsupervised Learning, Biomarker Discovery, Speech Analysis, Clustering, Mental Health

---

## 1. Introduction

### 1.1 Background

Major Depressive Disorder (MDD) is one of the most prevalent mental health conditions globally, affecting over 280 million people worldwide (WHO, 2021). Despite its significant impact on quality of life, work productivity, and mortality risk, MDD diagnosis remains predominantly subjective, relying on clinical interviews and self-report questionnaires such as the Patient Health Questionnaire (PHQ-9) and Beck Depression Inventory (BDI).

### 1.2 The Problem

Current diagnostic methods face several critical limitations:

1. **Subjectivity:** Diagnosis depends heavily on patient self-report and clinician interpretation
2. **Heterogeneity:** Depression manifests differently across individuals, yet treatment is often standardized
3. **Latency:** Symptoms must be present for weeks before diagnosis
4. **Stigma:** Many patients underreport symptoms due to social stigma

### 1.3 Objective Biomarkers: The Solution

Recent advances in signal processing and machine learning offer promising alternatives through **objective biomarkers** extracted from:

- **Acoustic signals:** Pitch, energy, speech rate, pause patterns
- **Linguistic patterns:** Word choice, sentiment, grammatical structure
- **Behavioral markers:** Response latency, emotional expression

### 1.4 Our Contribution

This study addresses the gap between subjective diagnosis and objective measurement by:

1. Applying **unsupervised machine learning** to discover hidden depression subtypes
2. Identifying **novel biomarker combinations** that characterize each subtype
3. Demonstrating that **acoustic and linguistic features** contain diagnostically relevant information
4. Providing a **replicable framework** for mental health biomarker discovery

### 1.5 Research Questions

1. Can unsupervised algorithms identify meaningful latent subtypes of MDD patients?
2. What acoustic and linguistic biomarkers define these subtypes?
3. Do discovered clusters correlate with clinical depression severity?
4. Can dimensionality reduction reveal interpretable emotional dimensions?

---

## 2. Literature Review

### 2.1 Depression and Mental Health AI

[Review 8-10 key papers on depression detection using ML]

- Gratch et al. (2014): DAIC-WOZ dataset for depression interviews
- Cummins et al. (2015): Speech analysis for depression assessment
- Alhanai et al. (2018): Context-aware depression detection from audio

### 2.2 Acoustic Biomarkers

[Review research on speech features in depression]

- Reduced pitch variability (prosody flattening)
- Increased pause duration and frequency
- Lower speech energy and tempo

### 2.3 Linguistic Biomarkers

[Review research on text features in depression]

- Increased first-person pronoun usage
- Higher negative emotion words
- Reduced cognitive complexity

### 2.4 Unsupervised Learning in Mental Health

[Review applications of clustering in psychiatry]

- Drysdale et al. (2017): fMRI-based depression subtypes
- Fried & Nesse (2015): Symptom network analysis

### 2.5 Research Gap

While supervised depression detection has been extensively studied, **unsupervised discovery of subtypes** from multimodal behavioral data remains underexplored. This study fills that gap.

---

## 3. Methodology

### 3.1 Dataset

**Dataset:** DAIC-WOZ Depression Database (Distress Analysis Interview Corpus)

- **Source:** USC Institute for Creative Technologies
- **Size:** 189 clinical interviews (107 with depression, 82 controls)
- **Modalities:** Audio recordings, video, text transcripts
- **Labels:** PHQ-8 depression severity scores (0-24 scale)
- **Duration:** 7-33 minutes per interview

**Data Split:**
- Training: 70%
- Validation: 15%
- Test: 15%

### 3.2 Preprocessing Pipeline

#### 3.2.1 Audio Preprocessing
1. Resample to 16 kHz
2. Remove silence (threshold: -20 dB)
3. Normalize amplitude
4. Segment into 3-second windows

#### 3.2.2 Text Preprocessing
1. Remove URLs, special characters
2. Remove filler words (um, uh, like)
3. Tokenization
4. Lemmatization
5. Remove stopwords (optional)

### 3.3 Feature Extraction

#### 3.3.1 Acoustic Features (89 features)

**MFCC Features (39):**
- 13 MFCCs + 13 deltas + 13 delta-deltas
- Statistics: mean, std, min, max

**Prosodic Features (15):**
- Pitch: mean, std, range, slope
- Energy: mean, std, dynamic range
- Tempo, zero-crossing rate
- Pause ratio, speech rate

**Spectral Features (35):**
- Spectral centroid, rolloff, bandwidth
- Spectral contrast (7 bands)
- Chroma features (12 pitch classes)

#### 3.3.2 Linguistic Features (42 features)

**Structural Features:**
- Word count, sentence count
- Type-token ratio
- Average word/sentence length

**Emotional Features:**
- Negative/positive word ratio
- First-person pronoun usage
- Death-related word count
- Sentiment polarity/subjectivity

**Cognitive Features:**
- Certainty/tentative word ratio
- Causation word count
- Negation frequency

**Speech-Specific Features:**
- Filler word ratio
- Pause indicators
- Repetition count

### 3.4 Multimodal Fusion

Features from all modalities were combined using:
- **Strategy:** Concatenation
- **Normalization:** StandardScaler (zero mean, unit variance)
- **Final dimension:** 131 features

### 3.5 Dimensionality Reduction

#### 3.5.1 PCA (Principal Component Analysis)
- **Purpose:** Remove noise, reduce redundancy
- **Configuration:** Preserve 95% variance
- **Result:** ~25-30 components

#### 3.5.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Purpose:** 2D visualization
- **Configuration:** perplexity=30, n_iter=1000
- **Result:** 2D embeddings

#### 3.5.3 UMAP (Uniform Manifold Approximation and Projection)
- **Purpose:** Fast nonlinear reduction
- **Configuration:** n_neighbors=15, min_dist=0.1
- **Result:** 2D embeddings

#### 3.5.4 VAE (Variational Autoencoder)
- **Purpose:** Learn latent emotional dimensions
- **Architecture:** [131 → 256 → 128 → 64 → 16 → 64 → 128 → 256 → 131]
- **Training:** 100 epochs, batch_size=32, lr=0.001
- **Result:** 16-dimensional latent space

### 3.6 Clustering Algorithms

#### 3.6.1 K-Means
- **Range:** k = 2, 3, 4, 5, 6
- **Initialization:** k-means++, n_init=50

#### 3.6.2 Gaussian Mixture Model (GMM)
- **Range:** n_components = 2, 3, 4, 5, 6
- **Covariance:** Full
- **Selection:** BIC/AIC criteria

#### 3.6.3 Spectral Clustering
- **Range:** k = 2, 3, 4, 5, 6
- **Affinity:** RBF kernel
- **Purpose:** Capture nonlinear patterns

### 3.7 Evaluation Metrics

**Clustering Quality:**
- Silhouette Score (higher = better)
- Davies-Bouldin Index (lower = better)
- Calinski-Harabasz Index (higher = better)

**Clinical Validation:**
- Correlation with PHQ-8 scores
- ANOVA tests between clusters
- Statistical significance (p < 0.05)

---

## 4. Results

### 4.1 Dimensionality Reduction Results

[Insert PCA variance plot]
[Insert t-SNE/UMAP scatter plots]
[Insert VAE latent space visualization]

### 4.2 Clustering Results

[Insert table comparing all clustering methods and metrics]

**Best Model:** [K-Means/GMM/Spectral] with **k = [X]** clusters
- Silhouette Score: [value]
- Davies-Bouldin: [value]

### 4.3 Discovered Subtypes

#### Cluster 1: [Name - e.g., "Cognitive Subtype"]
- **Size:** X patients (Y%)
- **PHQ-8 Score:** μ = [mean], σ = [std]
- **Key Biomarkers:**
  - Low pitch variability
  - High pause ratio
  - Elevated first-person pronoun use
  - Low speech energy

#### Cluster 2: [Name - e.g., "Emotional Exhaustion"]
[Similar breakdown]

### 4.4 Biomarker Analysis

[Insert heatmap of feature means per cluster]
[Insert radar chart comparing clusters]

**Top 10 Discriminative Features:**
1. [Feature name]: [importance score]
2. ...

### 4.5 Clinical Correlation

[Insert box plots of PHQ-8 scores by cluster]
[Insert statistical test results]

**Key Findings:**
- Significant differences between clusters (ANOVA p < 0.001)
- Cluster [X] had highest mean PHQ-8 score
- [Additional findings]

---

## 5. Discussion

### 5.1 Interpretation of Subtypes

[Interpret each cluster in clinical context]

### 5.2 Novel Biomarker Combinations

[Discuss unexpected feature combinations]

### 5.3 Comparison with Existing Literature

[Compare with other subtyping studies]

### 5.4 Clinical Implications

1. **Personalized Treatment:** Different subtypes may respond to different therapies
2. **Early Detection:** Objective biomarkers enable screening
3. **Progress Monitoring:** Track changes over time

### 5.5 Limitations

1. **Sample Size:** Limited to DAIC-WOZ dataset
2. **Cross-sectional:** No longitudinal data
3. **Modalities:** Missing neuroimaging data
4. **Validation:** Requires clinical validation

### 5.6 Future Work

1. Validate on independent datasets
2. Include neuroimaging modalities
3. Develop predictive models
4. Conduct longitudinal studies

---

## 6. Conclusion

This study successfully demonstrated that unsupervised machine learning can discover meaningful depression subtypes from multimodal behavioral data. We identified **[X]** distinct subtypes, each characterized by unique acoustic and linguistic biomarker patterns. These findings:

1. Provide **objective, measurable** markers for depression
2. Reveal **hidden heterogeneity** in MDD presentation
3. Enable **data-driven personalization** of treatment
4. Open new avenues for **computational psychiatry**

The fusion of speech analysis, natural language processing, and unsupervised learning represents a promising frontier in mental health diagnosis—moving from subjective assessment to objective, reproducible measurement.

---

## 7. References

[IEEE Format - Add 20-30 references]

[1] J. Gratch et al., "The Distress Analysis Interview Corpus of human and computer interviews," in *Proc. LREC*, 2014.

[2] N. Cummins et al., "A review of depression and suicide risk assessment using speech analysis," *Speech Communication*, vol. 71, pp. 10-49, 2015.

[3] A. T. Drysdale et al., "Resting-state connectivity biomarkers define neurophysiological subtypes of depression," *Nature Medicine*, vol. 23, no. 1, pp. 28-38, 2017.

[Continue with all references...]

---

## Appendix A: Detailed Feature List

[Table of all 131 features with descriptions]

## Appendix B: Statistical Tests

[Detailed statistical analysis results]

## Appendix C: Code Availability

Source code available at: [GitHub repository URL]

---

**Author Information:**
- **Name:** [Your Name]
- **Institution:** [Your University]
- **Email:** [Your Email]
- **Date:** November 2025

**Acknowledgments:**
We thank the USC Institute for Creative Technologies for providing the DAIC-WOZ dataset.

**Ethics Statement:**
This research uses publicly available, de-identified data. All participants in the original dataset provided informed consent.

**Conflict of Interest:**
The authors declare no conflicts of interest.
