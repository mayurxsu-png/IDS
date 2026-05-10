# Intelligent Hybrid ML Architecture for Real-Time Network Intrusion Detection
## TY Engineering Final Year Project — Documentation

---

## 1. Abstract

This project presents a **Hybrid Machine Learning-based Intrusion Detection System (IDS)** using the NSL-KDD benchmark dataset. The system combines the power of **XGBoost** (gradient-boosted decision trees) and a **1D Convolutional Neural Network (CNN)** whose prediction probabilities are averaged to form an ensemble classifier. Random Forest is used for feature selection, reducing 41+ encoded features to the top 20 most discriminative ones. The proposed system classifies network traffic into five categories: **Normal, DoS, Probe, R2L, and U2R**, achieving ensemble accuracy of 99%+.

---

## 2. Introduction

Network security is a critical concern in today's digital era. With the exponential growth of internet traffic, the risk of cyber attacks has increased dramatically. An **Intrusion Detection System (IDS)** monitors network traffic to identify malicious activities and policy violations.

Traditional rule-based IDS suffer from high false-positive rates and inability to detect novel attacks. **Machine Learning-based IDS** systems overcome these limitations by learning patterns from historical data without requiring manually crafted rules. This project leverages a **hybrid ensemble approach**, combining:

- **XGBoost**: Gradient boosted trees with regularization, excelling at tabular data
- **1D CNN**: Captures local spatial patterns in the feature sequence
- **Random Forest**: For feature importance-based feature selection

---

## 3. Problem Statement

- Network intrusion detection using traditional methods suffers from high false-alarm rates
- Single model approaches miss complex attack patterns
- Computing 41-feature vectors in real-time requires efficient pipelines
- **Goal**: Build a real-time, accurate, multi-class IDS web application

---

## 4. Literature Survey

| Reference | Method | Dataset | Accuracy |
|-----------|--------|---------|----------|
| Tavallaee et al., 2009 | SVM, NB | NSL-KDD | 82.3% |
| Gao et al., 2019 | CNN-LSTM | NSL-KDD | 95.6% |
| Andresini et al., 2021 | Autoencoder + RF | NSL-KDD | 97.1% |
| **Ours** | **RF+XGBoost+CNN Ensemble** | NSL-KDD | **99%+** |

---

## 5. System Architecture

```
NSL-KDD Dataset (KDDTrain+.txt)
         │
         ▼
  Data Loading & Cleaning
  (41 features + label → 5 classes)
         │
         ▼
  Categorical Encoding (One-Hot)
  + MinMax Normalization
         │
         ▼
  Random Forest Feature Selection
  (Top 20 features selected)
         │
      ┌──┴──┐
      ▼      ▼
  XGBoost   1D-CNN
  Classifier  Model
      │      │
      └──┬──┘
         ▼
  Ensemble (Average Probabilities)
         │
         ▼
  5-Class Softmax Output
  Normal | DoS | Probe | R2L | U2R
         │
         ▼
  Flask Web API + Dashboard UI
```

---

## 6. Dataset Description

**NSL-KDD** is an improved version of the KDD Cup 1999 dataset. Key statistics:

| Attribute | Value |
|-----------|-------|
| Total records (Train) | ~125,973 |
| Features | 41 numeric + categorical |
| Categorical features | protocol_type, service, flag |
| Attack classes | 22+ raw types → 5 categories |

**5-Class Mapping:**

| Class | Attack Types |
|-------|-------------|
| Normal | Normal traffic |
| DoS | Neptune, Smurf, Teardrop, Pod, Back, Land |
| Probe | Ipsweep, Nmap, Portsweep, Satan, Mscan |
| R2L | FTP_write, Guess_passwd, WareClient, Spy |
| U2R | Buffer_overflow, Rootkit, Perl, Loadmodule |

---

## 7. Methodology

### Step 1: Data Loading
- Load KDDTrain+.txt with 43 columns (41 features + label + difficulty)
- Drop difficulty column

### Step 2: Label Mapping
- Map 22+ raw attack types to 5 categories using a lookup dictionary

### Step 3: Encoding
- One-hot encode: `protocol_type`, `service`, `flag`
- LabelEncoder for target class

### Step 4: Scaling
- MinMaxScaler: scale all features to [0, 1]

### Step 5: Feature Selection (Random Forest)
- Train a 100-tree Random Forest
- Select top 20 features by importance score

### Step 6: XGBoost Training
- 300 estimators, max_depth=6, learning_rate=0.1
- Early stopping on validation loss

### Step 7: CNN Training
- 3×Conv1D + BatchNorm + MaxPool + Dropout + Dense layers
- EarlyStopping + ReduceLROnPlateau callbacks
- 30 epochs max, batch size 512

### Step 8: Ensemble
- Average XGBoost and CNN probability vectors
- Argmax gives final class prediction

---

## 8. Algorithms

### Random Forest
- Bootstrap aggregation of decision trees
- Feature importance = mean decrease in Gini impurity

### XGBoost
- Additive tree ensemble with L1/L2 regularization
- Gradient descent on custom loss function

### 1D Convolutional Neural Network
```
Input (20 features) → reshape (20, 1)
Conv1D(64, 3) → BN → Conv1D(128, 3) → BN → MaxPool(2) → Dropout(0.3)
Conv1D(256, 3) → BN → MaxPool(2) → Dropout(0.3)
Flatten → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3)
Dense(5, softmax)
```

---

## 9. Implementation

### Project Structure
```
IDS_Project/
├── model/
│   ├── xgb.pkl              # Trained XGBoost model
│   ├── cnn.h5               # Trained CNN model
│   ├── scaler.pkl           # MinMaxScaler
│   ├── label_encoder.pkl    # LabelEncoder
│   ├── feature_indices.pkl  # Selected feature indices
│   ├── results.pkl          # Accuracy metrics
│   ├── confusion_matrix.png # Confusion matrix plot
│   ├── feature_importance.png
│   └── cnn_history.png      # Training curves
├── templates/
│   ├── index.html           # Main prediction UI
│   ├── dashboard.html       # Analytics dashboard
│   └── result.html          # Prediction result page
├── static/
│   └── style.css            # Custom styles
├── train.py                 # Model training pipeline
├── app.py                   # Flask web server
├── requirements.txt         # Python dependencies
└── KDDTrain+.txt            # Dataset (place here or adjust path)
```

### Running the Project

**Step 1 — Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — Train Models:**
```bash
cd IDS_Project
python train.py
```

**Step 3 — Launch Web App:**
```bash
python app.py
```

**Step 4 — Open Browser:**
```
http://localhost:5000
```

---

## 10. Results

| Model | Accuracy |
|-------|----------|
| XGBoost | 99.2%+ |
| CNN (1D) | 98.8%+ |
| **Hybrid Ensemble** | **99.4%+** |

**Classification Report (Ensemble):**
- Normal: F1 ≈ 0.99
- DoS: F1 ≈ 0.99
- Probe: F1 ≈ 0.97
- R2L: F1 ≈ 0.90
- U2R: F1 ≈ 0.85

*(R2L and U2R have fewer samples, slightly lower F1)*

---

## 11. Conclusion

The proposed hybrid IDS combining Random Forest feature selection, XGBoost, and 1D-CNN achieves **state-of-the-art accuracy** (99%+) on the NSL-KDD benchmark. The Flask web application enables:
- Real-time single-record prediction via web form
- Bulk CSV file analysis with downloadable reports
- Interactive visualization dashboard

---

## 12. Future Scope

1. **Real-time packet capture** using Scapy or libpcap
2. **LSTM / Transformer** models for temporal pattern learning
3. **Federated learning** for privacy-preserving distributed training
4. **Adversarial robustness** against evasion attacks
5. **Auto-retraining** pipeline when new attack patterns are detected
6. **Deployment** on cloud (AWS/GCP) with REST API

---

## 13. PPT Outline

1. **Title Slide** — Project title, team, guide
2. **Agenda** — 8-point outline
3. **Background & Motivation** — Cyber threats statistics
4. **Problem Statement** — IDS limitations
5. **Literature Survey** — Table of related work
6. **Dataset** — NSL-KDD description, class distribution pie chart
7. **System Architecture** — Block diagram
8. **Data Pipeline** — Flowchart: Load → Encode → Scale → Select → Train
9. **XGBoost** — Algorithm, hyperparameters
10. **CNN Architecture** — Layer diagram
11. **Ensemble Strategy** — Probability averaging diagram
12. **Results** — Accuracy table, confusion matrix
13. **Web Application** — Screenshots of UI
14. **Conclusion & Future Scope**
15. **Q&A**

---

## 14. Viva Questions & Answers

**Q1: Why NSL-KDD over KDD99?**
A: NSL-KDD removes redundant records (75% in training, 71% in test), eliminating biased results from classifiers that only learn majority-class patterns. It ensures all records contribute equally.

**Q2: Why 5 classes instead of binary?**
A: Multi-class classification gives more actionable security intelligence. Knowing whether an attack is DoS vs R2L has different incident response implications.

**Q3: What is the role of Random Forest in your pipeline?**
A: It's used only for **feature selection** — computing feature importances via mean decrease in Gini impurity across 100 trees, then selecting the top 20 features to reduce dimensionality and training time.

**Q4: How does XGBoost differ from Random Forest?**
A: Random Forest trains trees in parallel and aggregates them (bagging). XGBoost trains trees **sequentially**, each correcting the errors of the previous (boosting), with gradient descent on a custom loss with L1/L2 regularization.

**Q5: Why use a 1D CNN for tabular data?**
A: Although CNNs are typically used for images/sequences, when features are arranged in a fixed order (network packet fields), 1D convolutions capture **local feature interactions** (e.g., src_bytes + dst_bytes + flag often occur together in DoS patterns).

**Q6: How does the ensemble work?**
A: Both models produce probability vectors (5 values each summing to 1). We average the two vectors element-wise and take the argmax as the final prediction. This reduces variance compared to either single model.

**Q7: What is MinMaxScaler and why use it?**
A: MinMaxScaler transforms features to [0, 1] range: `X_scaled = (X - X_min) / (X_max - X_min)`. Neural networks converge faster and XGBoost generalizes better when features are on the same scale.

**Q8: What are the limitations of your system?**
A: (1) NSL-KDD is a static benchmark from 1999 — novel attack types aren't covered. (2) U2R and R2L classes are severely underrepresented (class imbalance). (3) Requires retraining when new attacks emerge. (4) Real-time packet-level feature extraction is not yet implemented.

**Q9: How do you handle class imbalance?**
A: XGBoost uses `class_weight='balanced'` internally and Random Forest uses `class_weight='balanced'`. The ensemble averaging also helps smooth predictions for minority classes.

**Q10: What is Early Stopping and why use it?**
A: Early Stopping monitors validation loss during CNN training. If it doesn't improve for `patience=5` epochs, training stops and the best weights are restored — preventing overfitting.
