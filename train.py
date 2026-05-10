

import os, sys, warnings, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix)
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers  import (Conv1D, MaxPooling1D, Flatten,
                                      Dense, Dropout, BatchNormalization)
from tensorflow.keras.utils   import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ─── 0. Paths ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, '..', 'KDDTrain+.txt')   # project root
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. Column Names (41 features + label + difficulty) ───────────────────────
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

# ─── 2. Attack-to-5-Class Mapping ─────────────────────────────────────────────
ATTACK_MAP = {
    'normal':    'Normal',
    # DoS
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS',
    'worm':'DoS','mailbomb':'DoS',
    # Probe
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe',
    'mscan':'Probe','saint':'Probe',
    # R2L
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L',
    'phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L',
    'xlock':'R2L','xsnoop':'R2L','snmpguess':'R2L','snmpgetattack':'R2L',
    'httptunnel':'R2L','sendmail':'R2L','named':'R2L',
    # U2R
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'ps':'U2R','xterm':'U2R','sqlattack':'U2R',
}

CLASS_ORDER = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# ─── 3. Load & Pre-process ────────────────────────────────────────────────────
def load_data(path):
    print(f"\n[1/6] Loading dataset from: {path}")
    df = pd.read_csv(path, header=None, names=COLUMNS)
    print(f"      Shape: {df.shape}")

    # Map labels → 5 classes
    df['label'] = df['label'].str.strip()
    df['attack_cat'] = df['label'].map(ATTACK_MAP)
    unmapped = df['attack_cat'].isna().sum()
    if unmapped:
        print(f"      WARNING: {unmapped} rows with unknown labels → mapping to 'Normal'")
        df['attack_cat'] = df['attack_cat'].fillna('Normal')

    df.drop(columns=['label','difficulty'], inplace=True)

    # Distribution
    print("\n      Attack category distribution:")
    for c in CLASS_ORDER:
        n = (df['attack_cat'] == c).sum()
        print(f"        {c:10s}: {n:>7,}  ({100*n/len(df):.2f}%)")
    return df

def preprocess(df):
    print("\n[2/6] Encoding categorical columns & scaling")
    # One-hot encode
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=False)

    # Encode target
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y = le.transform(df['attack_cat'])
    X = df.drop(columns=['attack_cat'])

    # Ensure numeric
    X = X.astype(np.float32)

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = X.columns.tolist()

    print(f"      Features after encoding: {X_scaled.shape[1]}")
    return X_scaled, y, le, scaler, feature_names

# ─── 4. Feature Selection via Random Forest ───────────────────────────────────
def select_features(X, y, feature_names, n_top=20):
    print(f"\n[3/6] Random Forest feature selection (top {n_top})")
    rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                n_jobs=-1, class_weight='balanced')
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:n_top]

    selected_names = [feature_names[i] for i in indices]
    print(f"      Top {n_top} features: {selected_names}")

    # Save importance plot
    plt.figure(figsize=(12, 6))
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.bar(range(n_top), importances[indices], color='steelblue')
    plt.xticks(range(n_top), selected_names, rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    return indices, selected_names

# ─── 5. Build 1D-CNN ──────────────────────────────────────────────────────────
def build_cnn(input_dim, n_classes):
    model = Sequential([
        tf.keras.Input(shape=(input_dim, 1)),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ─── 6. Train & Evaluate ──────────────────────────────────────────────────────
def train_evaluate(X_sel, y, le, indices, selected_names):
    n_classes = len(CLASS_ORDER)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sel, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n[4/6] Train: {X_tr.shape[0]:,}  |  Test: {X_te.shape[0]:,}")

    # ── 4a. XGBoost ─────────────────────────────────────────────────────────
    print("\n      Training XGBoost …")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1,
        early_stopping_rounds=10,
    )
    xgb_model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  verbose=False)
    xgb_preds  = xgb_model.predict(X_te)
    xgb_proba  = xgb_model.predict_proba(X_te)
    xgb_acc    = accuracy_score(y_te, xgb_preds)
    print(f"      XGBoost  accuracy : {xgb_acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    with open(os.path.join(MODEL_DIR, 'xgb.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)

    # ── 4b. 1D-CNN ──────────────────────────────────────────────────────────
    print("\n      Training 1D-CNN …")
    t0 = time.time()
    X_tr_cnn = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
    X_te_cnn = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)
    y_tr_oh  = to_categorical(y_tr, n_classes)
    y_te_oh  = to_categorical(y_te, n_classes)

    cnn_model = build_cnn(X_tr.shape[1], n_classes)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    ]
    history = cnn_model.fit(
        X_tr_cnn, y_tr_oh,
        epochs=15, batch_size=512,
        validation_data=(X_te_cnn, y_te_oh),
        callbacks=callbacks, verbose=1,
    )

    cnn_proba  = cnn_model.predict(X_te_cnn, verbose=0)
    cnn_preds  = np.argmax(cnn_proba, axis=1)
    cnn_acc    = accuracy_score(y_te, cnn_preds)
    print(f"      CNN      accuracy : {cnn_acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    cnn_model.save(os.path.join(MODEL_DIR, 'cnn.h5'))

    # ── 4c. Ensemble (average probabilities) ────────────────────────────────
    ens_proba  = (xgb_proba + cnn_proba) / 2
    ens_preds  = np.argmax(ens_proba, axis=1)
    ens_acc    = accuracy_score(y_te, ens_preds)
    print(f"\n      Ensemble accuracy : {ens_acc*100:.2f}%")

    # ── 5. Metrics & Plots ──────────────────────────────────────────────────
    class_names = le.classes_
    print("\n[5/6] Classification Report (Ensemble):")
    print(classification_report(y_te, ens_preds, target_names=class_names))

    cm = confusion_matrix(y_te, ens_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix – Hybrid Ensemble')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Training history plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('CNN Accuracy'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('CNN Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'cnn_history.png'), dpi=150)
    plt.close()

    # Accuracy summary
    results = {
        'xgb_accuracy':      round(xgb_acc * 100, 2),
        'cnn_accuracy':      round(cnn_acc * 100, 2),
        'ensemble_accuracy': round(ens_acc * 100, 2),
        'class_names':       list(class_names),
        'n_features':        X_sel.shape[1],
        'selected_features': selected_names,
    }
    with open(os.path.join(MODEL_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"\n[6/6] All models & artefacts saved to: {MODEL_DIR}")
    return results

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  IDS Hybrid ML Training Pipeline")
    print("=" * 65)

    df = load_data(DATA_FILE)
    X, y, le, scaler, feature_names = preprocess(df)

    # Save encoder & scaler
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    indices, selected_names = select_features(X, y, feature_names, n_top=20)
    X_sel = X[:, indices]

    # Save selected feature indices
    with open(os.path.join(MODEL_DIR, 'feature_indices.pkl'), 'wb') as f:
        pickle.dump({'indices': indices, 'names': selected_names,
                     'all_features': feature_names}, f)

    results = train_evaluate(X_sel, y, le, indices, selected_names)

    print("\n" + "=" * 65)
    print(f"  XGBoost  Accuracy : {results['xgb_accuracy']}%")
    print(f"  CNN      Accuracy : {results['cnn_accuracy']}%")
    print(f"  Ensemble Accuracy : {results['ensemble_accuracy']}%")
    print("=" * 65)

if __name__ == '__main__':
    main()
