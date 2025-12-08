# 🎯 DAIC-WOZ Depression Database Setup Guide

## 📊 Dataset Overview

**DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)**
- **189 clinical interview sessions** (IDs: 300-492)
- **Multimodal data**: Audio, video, transcripts, facial features, acoustic features
- **Labels**: PHQ-8 depression scores and binary classification (PHQ-8 ≥ 10 = depressed)
- **Official splits**: Train/Dev/Test provided by AVEC 2017

## 📁 Recommended Folder Structure

```
m:\5th sem\ML2-project\
├── data/
│   ├── raw/                          # Raw downloaded sessions
│   │   ├── 300_P/
│   │   ├── 301_P/
│   │   └── ... (189 folders)
│   │
│   ├── splits/                       # Official train/dev/test splits
│   │   ├── train_split_Depression_AVEC2017.csv
│   │   ├── dev_split_Depression_AVEC2017.csv
│   │   ├── test_split_Depression_AVEC2017.csv
│   │   └── full_test_split.csv
│   │
│   ├── processed/                    # Processed features
│   │   ├── train/
│   │   ├── dev/
│   │   └── test/
│   │
│   └── documentation/
│       ├── DAICWOZDepression_Documentation_AVEC2017.pdf
│       └── documents.zip (extracted)
```

## 📥 Step 1: Download Essential Files

### Critical CSV Files (Download First - Small Size):
```
Priority 1 - Labels & Splits:
✓ train_split_Depression_AVEC2017.csv (3.0K) 
✓ dev_split_Depression_AVEC2017.csv (1.1K)
✓ test_split_Depression_AVEC2017.csv (304 bytes)
✓ full_test_split.csv (620 bytes)

Priority 2 - Documentation:
✓ DAICWOZDepression_Documentation_AVEC2017.pdf (92K)
✓ documents.zip (5.8M)
✓ util.zip (1.2K)
```

### Where to Place CSV Files:
```powershell
# Create the splits folder
New-Item -ItemType Directory -Path "data/splits" -Force

# Place the CSV files here:
data/splits/train_split_Depression_AVEC2017.csv
data/splits/dev_split_Depression_AVEC2017.csv  
data/splits/test_split_Depression_AVEC2017.csv
data/splits/full_test_split.csv
```

## 📥 Step 2: Download Session Data (Gradual)

### Strategy: Start with Training Set Only

**Training set contains ~107 sessions**. Start by downloading **only train split sessions** to save space and time.

### Recommended Download Approach:

**Option A: Download Top Priority Sessions (Smaller files first)**
```
Start with sessions < 400MB for faster testing:
- 300_P.zip (327M)
- 301_P.zip (403M)
- 309_P.zip (346M)
- 318_P.zip (287M)
- 319_P.zip (310M)
- 357_P.zip (187M)
... (download based on train_split.csv)
```

**Option B: Download ALL Training Sessions**
- Check `train_split_Depression_AVEC2017.csv` for session IDs
- Download corresponding `XXX_P.zip` files
- Extract to `data/raw/`

### Total Dataset Size:
- **All 189 sessions**: ~85-90 GB
- **Train only (~107 sessions)**: ~50-55 GB
- **Test 10 sessions for testing**: ~4-5 GB

## 📂 Step 3: Extract and Organize

```powershell
# Extract each session
Expand-Archive -Path "300_P.zip" -DestinationPath "data/raw/300_P"

# Each session contains:
300_P/
├── 300_AUDIO.wav              # 16kHz audio recording
├── 300_TRANSCRIPT.csv         # Interview transcript
├── 300_COVAREP.csv           # Acoustic features (F0, MFCC, etc.)
├── 300_FORMANT.csv           # Formant frequencies
├── 300_CLNF_AUs.csv          # Facial Action Units
├── 300_CLNF_features.txt     # 2D facial landmarks
├── 300_CLNF_features3D.txt   # 3D facial landmarks
├── 300_CLNF_gaze.txt         # Eye gaze direction
├── 300_CLNF_pose.txt         # Head pose
└── 300_CLNF_hog.bin          # HOG features (binary)
```

## 🎯 Step 4: What to Use for Your Project

### For Unsupervised Learning (Current Project):

**Focus on these modalities:**

1. **Text/Transcript Features** ✅ (Already implemented in notebook)
   - `XXX_TRANSCRIPT.csv` - Interview transcripts
   - Extract linguistic biomarkers (you already have this code!)

2. **Audio/Acoustic Features** 🎵 (Easy to add)
   - `XXX_COVAREP.csv` - 74 acoustic features per 10ms
   - `XXX_FORMANT.csv` - Formant frequencies
   - Features: F0, MFCC, NAQ, jitter, shimmer, etc.

3. **Facial Features** 👤 (Optional - advanced)
   - `XXX_CLNF_AUs.csv` - Action Units (facial expressions)
   - Useful for depression detection (lack of facial expression)

### Excluded Sessions (Don't Download):
- 342, 394, 398, 460 (technical issues)

## 📊 CSV File Formats

### train_split_Depression_AVEC2017.csv
```csv
Participant_ID, PHQ8_Binary, PHQ8_Score, Gender, PHQ_Q1, PHQ_Q2, ..., PHQ_Q8
300, 0, 3, Male, 0, 1, 0, 1, 0, 0, 1, 0
301, 1, 15, Female, 2, 3, 2, 2, 2, 1, 2, 1
...
```
- **PHQ8_Binary**: 0=No Depression, 1=Depression (PHQ-8 ≥ 10)
- **PHQ8_Score**: Total score (0-24)
- **Gender**: Male/Female

## 🚀 Quick Start Workflow

### Minimal Setup (Test First):

```powershell
# 1. Download small files
cd "m:\5th sem\ML2-project\data"
New-Item -ItemType Directory -Path "splits" -Force
New-Item -ItemType Directory -Path "raw" -Force

# 2. Download CSV splits to data/splits/
# (Use browser or wget/curl if available)

# 3. Download 5-10 small sessions for testing (e.g., 300-310)
# Extract to data/raw/

# 4. Run the new DAIC-WOZ notebook we'll create
```

## 💡 Project Integration Options

### Option 1: Keep Current Kaggle Dataset + Add DAIC-WOZ
- Continue with Mental Health Text dataset (quick results)
- Add DAIC-WOZ analysis as **extended/advanced section**
- Compare unsupervised findings across datasets

### Option 2: Full DAIC-WOZ Pipeline (Gold Standard)
- Use official train/dev/test splits
- Multimodal features (audio + text + facial)
- Directly comparable to AVEC 2017 benchmark papers
- **Best for academic publication**

## 📝 Next Steps

1. **Download CSV splits** → Place in `data/splits/`
2. **Download 5-10 training sessions** → Test extraction and loading
3. **I'll create a new notebook**: `03_DAIC_WOZ_analysis.ipynb`
4. **Choose integration strategy**: Keep both datasets or focus on DAIC-WOZ

## 📚 Key Files for Features

| File | Size | Priority | Use Case |
|------|------|----------|----------|
| `train_split.csv` | 3KB | **CRITICAL** | Labels, training IDs |
| `dev_split.csv` | 1KB | **CRITICAL** | Validation labels |
| `XXX_TRANSCRIPT.csv` | Small | **HIGH** | Text features (current pipeline) |
| `XXX_COVAREP.csv` | Medium | **HIGH** | Acoustic features (89 features) |
| `XXX_AUDIO.wav` | Large | Medium | Raw audio (if processing from scratch) |
| `XXX_CLNF_AUs.csv` | Medium | Optional | Facial expressions |

## 🎓 Academic Advantages

Using DAIC-WOZ gives you:
- ✅ Standardized benchmark dataset
- ✅ Official train/dev/test splits (no data leakage)
- ✅ Comparable to published AVEC 2017 papers
- ✅ Multimodal features (audio + text + visual)
- ✅ Clinical labels (PHQ-8 scores)
- ✅ Much stronger for thesis/publication

---

**Ready to proceed?** Let me know:
1. Do you want to download the CSV files now?
2. Should I create a DAIC-WOZ specific notebook?
3. Keep both datasets or switch entirely to DAIC-WOZ?
