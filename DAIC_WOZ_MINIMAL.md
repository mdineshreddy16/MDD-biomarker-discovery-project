# DAIC-WOZ Dataset - Minimal Setup

## 📊 What You Need

**189 clinical interviews** (sessions 300-492)
- **Labels**: PHQ-8 depression scores (0=healthy, 1=depressed if score ≥10)
- **Data**: Audio transcripts + acoustic features
- **Splits**: Official train/dev/test

## 📥 Download (Minimum Required)

### 1. CSV Files (5 KB total - CRITICAL)
```
http://dcapswoz.ict.usc.edu/wwwdaicwoz/train_split_Depression_AVEC2017.csv
http://dcapswoz.ict.usc.edu/wwwdaicwoz/dev_split_Depression_AVEC2017.csv
http://dcapswoz.ict.usc.edu/wwwdaicwoz/test_split_Depression_AVEC2017.csv
```
**Save to**: `data/splits/`

### 2. Session Data (Start Small)
Download 10-20 training sessions first (~5-8 GB):
```
http://dcapswoz.ict.usc.edu/wwwdaicwoz/300_P.zip (327M)
http://dcapswoz.ict.usc.edu/wwwdaicwoz/301_P.zip (403M)
...
```
**Extract to**: `data/raw/300_P/`, `data/raw/301_P/`, etc.

## 📁 Folder Structure

```
data/
├── splits/
│   ├── train_split_Depression_AVEC2017.csv  ← IDs + labels
│   ├── dev_split_Depression_AVEC2017.csv
│   └── test_split_Depression_AVEC2017.csv
│
└── raw/
    ├── 300_P/
    │   ├── 300_TRANSCRIPT.csv     ← Use this (text)
    │   └── 300_COVAREP.csv        ← Use this (74 acoustic features)
    ├── 301_P/
    └── ...
```

## 🎯 What to Use

**For your unsupervised learning project:**
1. **Text**: `XXX_TRANSCRIPT.csv` - Interview transcripts
2. **Audio**: `XXX_COVAREP.csv` - 74 acoustic features (F0, MFCC, jitter, etc.)

Ignore video/facial files for now (optional later).

## ⚡ Quick Start

```powershell
# 1. Create folders
cd "m:\5th sem\ML2-project"
New-Item -ItemType Directory -Path "data\splits", "data\raw" -Force

# 2. Download CSV files manually to data/splits/

# 3. Download 10 sessions
# Download 300_P.zip through 309_P.zip from URL above
# Extract each to data/raw/

# 4. Run notebook
jupyter notebook notebooks/03_DAICWOZ_analysis.ipynb
```

## 📊 CSV Format

**train_split_Depression_AVEC2017.csv**:
```
Participant_ID,PHQ8_Binary,PHQ8_Score,Gender
300,0,3,Male
301,1,15,Female
...
```
- PHQ8_Binary: 0=No depression, 1=Depression
- PHQ8_Score: 0-24 (≥10 = depression threshold)

## 💾 Storage

- **Minimal (10 sessions)**: ~4 GB
- **Training set only**: ~50 GB
- **Full dataset**: ~85 GB

Start with 10-20 sessions, then download more as needed.
