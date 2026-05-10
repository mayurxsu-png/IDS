

import os, pickle, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
TEST_FILE = os.path.join(BASE_DIR, 'KDDTest+.txt')

# ─── Column definitions ──────────────────────────────────────────────────────
COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty'
]

CATEGORICAL = ['protocol_type', 'service', 'flag']

ATTACK_MAP = {
    'normal':    'Normal',
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS',
    'worm':'DoS','mailbomb':'DoS',
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe',
    'mscan':'Probe','saint':'Probe',
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L',
    'phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L',
    'xlock':'R2L','xsnoop':'R2L','snmpguess':'R2L','snmpgetattack':'R2L',
    'httptunnel':'R2L','sendmail':'R2L','named':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'ps':'U2R','xterm':'U2R','sqlattack':'U2R',
}

CLASS_ORDER = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


# ─── 1. Load all saved models & artifacts ─────────────────────────────────────
def load_models():
    """Load all trained models and preprocessing artifacts."""
    print("\n[1/5] Loading models from:", MODEL_DIR)
    models = {}

    with open(os.path.join(MODEL_DIR, 'xgb.pkl'), 'rb') as f:
        models['xgb'] = pickle.load(f)
    print("      ✓ XGBoost loaded")

    models['cnn'] = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'cnn.h5'))
    print("      ✓ CNN loaded")

    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        models['scaler'] = pickle.load(f)
    print("      ✓ Scaler loaded")

    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        models['le'] = pickle.load(f)
    print("      ✓ Label encoder loaded")

    with open(os.path.join(MODEL_DIR, 'feature_indices.pkl'), 'rb') as f:
        feat = pickle.load(f)
        models['indices']      = feat['indices']
        models['all_features'] = feat['all_features']
        models['sel_features'] = feat['names']
    print(f"      ✓ Feature indices loaded ({len(models['sel_features'])} selected features)")

    return models


# ─── 2. Load & preprocess test data ──────────────────────────────────────────
def load_test_data(path):
    """Load KDDTest+ and map labels to 5-class categories."""
    print(f"\n[2/5] Loading test data from: {path}")
    df = pd.read_csv(path, header=None, names=COLUMNS)
    print(f"      Shape: {df.shape}")

    # Map labels → 5 classes
    df['label'] = df['label'].str.strip()
    df['attack_cat'] = df['label'].map(ATTACK_MAP)
    unmapped = df['attack_cat'].isna().sum()
    if unmapped:
        print(f"      WARNING: {unmapped} rows with unknown labels → mapping to 'Normal'")
        df['attack_cat'] = df['attack_cat'].fillna('Normal')

    # Show distribution
    print("\n      Test set distribution:")
    for c in CLASS_ORDER:
        n = (df['attack_cat'] == c).sum()
        print(f"        {c:10s}: {n:>7,}  ({100*n/len(df):.2f}%)")

    return df


def preprocess_test(df, models):
    """Encode, scale, and select features — matching training pipeline exactly."""
    print("\n[3/5] Preprocessing test data ...")

    le = models['le']
    le.fit(CLASS_ORDER)
    y_true = le.transform(df['attack_cat'])

    # Drop non-feature columns
    df_feat = df.drop(columns=['label', 'difficulty', 'attack_cat'])

    # One-hot encode categorical
    df_enc = pd.get_dummies(df_feat, columns=CATEGORICAL, drop_first=False)

    # Align to training feature space
    df_enc = df_enc.reindex(columns=models['all_features'], fill_value=0).astype(np.float32)

    # Scale
    X_scaled = models['scaler'].transform(df_enc)

    # Select top features
    X_sel = X_scaled[:, models['indices']]

    print(f"      Features after selection: {X_sel.shape[1]}")
    print(f"      Total test samples: {X_sel.shape[0]:,}")
    return X_sel, y_true


# ─── 3. Run predictions for each model ───────────────────────────────────────
def test_all_models(X_sel, y_true, models):
    """Test XGBoost, CNN, and Ensemble. Print accuracy & classification reports."""
    le = models['le']
    class_names = le.classes_
    n_classes = len(class_names)

    results = {}

    print("\n" + "=" * 70)
    print("  MODEL EVALUATION ON KDDTest+ DATASET")
    print("=" * 70)

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODEL 1: XGBoost                                              │")
    print("└─────────────────────────────────────────────────────────────────┘")
    t0 = time.time()
    xgb_preds  = models['xgb'].predict(X_sel)
    xgb_proba  = models['xgb'].predict_proba(X_sel)
    xgb_time   = time.time() - t0
    xgb_acc    = accuracy_score(y_true, xgb_preds)
    print(f"  Accuracy : {xgb_acc*100:.2f}%")
    print(f"  Time     : {xgb_time:.2f}s")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, xgb_preds, target_names=class_names, digits=4))
    results['xgb'] = {
        'accuracy': xgb_acc, 'preds': xgb_preds,
        'proba': xgb_proba, 'time': xgb_time
    }

    # ── CNN ──────────────────────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODEL 2: 1D-CNN                                               │")
    print("└─────────────────────────────────────────────────────────────────┘")
    X_cnn = X_sel.reshape(X_sel.shape[0], X_sel.shape[1], 1)
    t0 = time.time()
    cnn_proba  = models['cnn'].predict(X_cnn, verbose=0)
    cnn_preds  = np.argmax(cnn_proba, axis=1)
    cnn_time   = time.time() - t0
    cnn_acc    = accuracy_score(y_true, cnn_preds)
    print(f"  Accuracy : {cnn_acc*100:.2f}%")
    print(f"  Time     : {cnn_time:.2f}s")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, cnn_preds, target_names=class_names, digits=4))
    results['cnn'] = {
        'accuracy': cnn_acc, 'preds': cnn_preds,
        'proba': cnn_proba, 'time': cnn_time
    }

    # ── Ensemble (0.6 XGB + 0.4 CNN) ────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODEL 3: Hybrid Ensemble (0.6×XGBoost + 0.4×CNN)              │")
    print("└─────────────────────────────────────────────────────────────────┘")
    t0 = time.time()
    ens_proba  = 0.6 * xgb_proba + 0.4 * cnn_proba
    ens_preds  = np.argmax(ens_proba, axis=1)
    ens_time   = xgb_time + cnn_time + (time.time() - t0)
    ens_acc    = accuracy_score(y_true, ens_preds)
    print(f"  Accuracy : {ens_acc*100:.2f}%")
    print(f"  Time     : {ens_time:.2f}s (combined)")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, ens_preds, target_names=class_names, digits=4))
    results['ensemble'] = {
        'accuracy': ens_acc, 'preds': ens_preds,
        'proba': ens_proba, 'time': ens_time
    }

    return results


# ─── 4. Per-class accuracy comparison ────────────────────────────────────────
def per_class_comparison(y_true, results, class_names):
    """Print a side-by-side per-class accuracy table."""
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  PER-CLASS ACCURACY COMPARISON                                  │")
    print("└─────────────────────────────────────────────────────────────────┘")
    header = f"  {'Class':10s} | {'XGBoost':>10s} | {'CNN':>10s} | {'Ensemble':>10s} | {'Support':>8s}"
    print(header)
    print("  " + "─" * len(header.strip()))

    for i, cls in enumerate(class_names):
        mask = (y_true == i)
        support = mask.sum()
        if support == 0:
            continue
        xgb_cls_acc = accuracy_score(y_true[mask], results['xgb']['preds'][mask]) * 100
        cnn_cls_acc = accuracy_score(y_true[mask], results['cnn']['preds'][mask]) * 100
        ens_cls_acc = accuracy_score(y_true[mask], results['ensemble']['preds'][mask]) * 100
        print(f"  {cls:10s} | {xgb_cls_acc:9.2f}% | {cnn_cls_acc:9.2f}% | {ens_cls_acc:9.2f}% | {support:>8,}")


# ─── 5. Sample predictions ──────────────────────────────────────────────────
def show_sample_predictions(X_sel, y_true, df, results, models, n=10):
    """Show sample predictions from each model."""
    le = models['le']
    class_names = le.classes_

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print(f"│  SAMPLE PREDICTIONS (first {n} rows)                             │")
    print("└─────────────────────────────────────────────────────────────────┘")

    header = f"  {'Row':>4s} | {'True Label':>12s} | {'XGBoost':>12s} | {'CNN':>12s} | {'Ensemble':>12s} | {'Match':>5s}"
    print(header)
    print("  " + "─" * len(header.strip()))

    for i in range(min(n, len(y_true))):
        true_label = class_names[y_true[i]]
        xgb_label  = class_names[results['xgb']['preds'][i]]
        cnn_label  = class_names[results['cnn']['preds'][i]]
        ens_label  = class_names[results['ensemble']['preds'][i]]
        match      = "✓" if ens_label == true_label else "✗"
        print(f"  {i+1:>4d} | {true_label:>12s} | {xgb_label:>12s} | {cnn_label:>12s} | {ens_label:>12s} | {match:>5s}")

    # Show one sample from each attack category
    print(f"\n  ── One sample per attack class ──")
    print(header)
    print("  " + "─" * len(header.strip()))

    for cls_idx, cls_name in enumerate(class_names):
        indices = np.where(y_true == cls_idx)[0]
        if len(indices) == 0:
            continue
        i = indices[0]
        true_label = cls_name
        xgb_label  = class_names[results['xgb']['preds'][i]]
        cnn_label  = class_names[results['cnn']['preds'][i]]
        ens_label  = class_names[results['ensemble']['preds'][i]]
        match      = "✓" if ens_label == true_label else "✗"
        orig_label = df['label'].iloc[i]
        print(f"  {i+1:>4d} | {true_label:>12s} | {xgb_label:>12s} | {cnn_label:>12s} | {ens_label:>12s} | {match:>5s}  ({orig_label})")


# ─── 6. Confidence analysis ─────────────────────────────────────────────────
def confidence_analysis(y_true, results, class_names):
    """Show average confidence for correct vs incorrect predictions."""
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  CONFIDENCE ANALYSIS (Ensemble)                                 │")
    print("└─────────────────────────────────────────────────────────────────┘")

    ens_proba = results['ensemble']['proba']
    ens_preds = results['ensemble']['preds']
    ens_conf  = np.max(ens_proba, axis=1) * 100

    correct_mask   = (ens_preds == y_true)
    incorrect_mask = ~correct_mask

    avg_correct   = ens_conf[correct_mask].mean() if correct_mask.any() else 0
    avg_incorrect = ens_conf[incorrect_mask].mean() if incorrect_mask.any() else 0

    print(f"  Avg confidence (correct predictions)   : {avg_correct:.2f}%")
    print(f"  Avg confidence (incorrect predictions)  : {avg_incorrect:.2f}%")
    print(f"  Total correct   : {correct_mask.sum():,} / {len(y_true):,}")
    print(f"  Total incorrect : {incorrect_mask.sum():,} / {len(y_true):,}")

    # Confusion matrix summary
    print(f"\n  Confusion Matrix (Ensemble):")
    cm = confusion_matrix(y_true, ens_preds)
    print(f"  {'':12s}", end="")
    for cls in class_names:
        print(f" {cls:>8s}", end="")
    print()
    for i, cls in enumerate(class_names):
        print(f"  {cls:12s}", end="")
        for j in range(len(class_names)):
            print(f" {cm[i][j]:>8d}", end="")
        print()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  IDS Model Testing & Prediction Script")
    print("  Testing: XGBoost | 1D-CNN | Hybrid Ensemble")
    print("=" * 70)

    # Load
    models = load_models()
    df = load_test_data(TEST_FILE)
    X_sel, y_true = preprocess_test(df, models)

    # Test all models
    print("\n[4/5] Running predictions on test set ...")
    results = test_all_models(X_sel, y_true, models)

    # Per-class comparison
    class_names = models['le'].classes_
    per_class_comparison(y_true, results, class_names)

    # Sample predictions
    print("\n[5/5] Sample predictions & analysis")
    show_sample_predictions(X_sel, y_true, df, results, models, n=15)

    # Confidence analysis
    confidence_analysis(y_true, results, class_names)

    # ── Final Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL ACCURACY SUMMARY")
    print("=" * 70)
    print(f"  XGBoost          : {results['xgb']['accuracy']*100:.2f}%  ({results['xgb']['time']:.2f}s)")
    print(f"  1D-CNN           : {results['cnn']['accuracy']*100:.2f}%  ({results['cnn']['time']:.2f}s)")
    print(f"  Hybrid Ensemble  : {results['ensemble']['accuracy']*100:.2f}%  ({results['ensemble']['time']:.2f}s)")
    print("=" * 70)

    best = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n  🏆 Best Model: {best.upper()} ({results[best]['accuracy']*100:.2f}%)")
    print()


if __name__ == '__main__':
    main()
