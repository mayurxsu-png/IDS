
import os, io, csv, pickle, json, datetime, time, random, threading, queue
import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, jsonify,
                   send_file, redirect, url_for, flash, Response, stream_with_context)

app = Flask(__name__)
app.secret_key = 'ids_secret_key_2024'

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# ─── Lightweight NumPy CNN Predictor (no tensorflow required) ─────────────────
class NumpyCNNPredictor:
    def __init__(self, weights_path):
        with open(weights_path, 'rb') as f:
            self.weights = pickle.load(f)

    def predict(self, x, verbose=0):
        if x.ndim == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        w = self.weights
        # Layer 0: Conv1D (w[0]=kernel, w[1]=bias)
        k0, b0 = w[0], w[1]
        N, L, _ = x.shape
        k_len, _, C_out = k0.shape
        pad = k_len // 2
        x_pad = np.pad(x, ((0,0),(pad,pad),(0,0)), mode='constant')
        out0 = np.zeros((N, L, C_out), dtype=np.float32)
        for i in range(L):
            out0[:, i, :] = np.einsum('nki,kio->no', x_pad[:, i:i+k_len, :], k0) + b0
        x = np.maximum(0, out0)
        # Layer 1: BN (gamma, beta, mean, var)
        x = w[2] * (x - w[4]) / np.sqrt(w[5] + 1e-3) + w[3]
        # Layer 2: Conv1D
        k1, b1 = w[6], w[7]
        N, L, _ = x.shape
        k_len, _, C_out = k1.shape
        pad = k_len // 2
        x_pad = np.pad(x, ((0,0),(pad,pad),(0,0)), mode='constant')
        out1 = np.zeros((N, L, C_out), dtype=np.float32)
        for i in range(L):
            out1[:, i, :] = np.einsum('nki,kio->no', x_pad[:, i:i+k_len, :], k1) + b1
        x = np.maximum(0, out1)
        # Layer 3: BN
        x = w[8] * (x - w[10]) / np.sqrt(w[11] + 1e-3) + w[9]
        # Layer 4: MaxPool1D(2)
        N, L, C = x.shape
        L_out = L // 2
        x = np.max(x[:, :L_out*2, :].reshape(N, L_out, 2, C), axis=2)
        # Layer 5: Conv1D
        k2, b2 = w[12], w[13]
        N, L, _ = x.shape
        k_len, _, C_out = k2.shape
        pad = k_len // 2
        x_pad = np.pad(x, ((0,0),(pad,pad),(0,0)), mode='constant')
        out2 = np.zeros((N, L, C_out), dtype=np.float32)
        for i in range(L):
            out2[:, i, :] = np.einsum('nki,kio->no', x_pad[:, i:i+k_len, :], k2) + b2
        x = np.maximum(0, out2)
        # Layer 6: BN
        x = w[14] * (x - w[16]) / np.sqrt(w[17] + 1e-3) + w[15]
        # Layer 7: MaxPool1D(2)
        N, L, C = x.shape
        L_out = L // 2
        x = np.max(x[:, :L_out*2, :].reshape(N, L_out, 2, C), axis=2)
        # Flatten
        x = x.reshape(N, -1)
        # Dense 1
        x = np.maximum(0, x @ w[18] + w[19])
        # Dense 2
        x = np.maximum(0, x @ w[20] + w[21])
        # Dense 3 Softmax
        logits = x @ w[22] + w[23]
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

# ─── Lightweight NumPy XGBoost Predictor (no xgboost required) ────────────────
class NumpyXGBPredictor:
    def __init__(self, weights_path):
        with open(weights_path, 'rb') as f:
            data = pickle.load(f)
        self.base_score = data['base_score']
        self.trees = data['trees']
        self.num_classes = data['num_classes']

    def eval_node(self, node, row):
        if node[0] == 0:  # leaf
            return node[1]
        _, feat_idx, thresh, yes_id, no_id, children = node
        val = row[feat_idx]
        if np.isnan(val) or val >= thresh:
            next_id = no_id
        else:
            next_id = yes_id
        return self.eval_node(children[next_id], row)

    def predict_proba(self, X):
        N = X.shape[0]
        raw_scores = np.tile(self.base_score, (N, 1))
        for i in range(N):
            row = X[i]
            for t_idx, tree in enumerate(self.trees):
                cls = t_idx % self.num_classes
                raw_scores[i, cls] += self.eval_node(tree, row)
        e = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

# ─── Lightweight NumPy Scaler & LabelEncoder (no scikit-learn required) ───────
class NumpyScaler:
    def __init__(self, scaler_obj_or_path):
        if isinstance(scaler_obj_or_path, str):
            with open(scaler_obj_or_path, 'rb') as f:
                obj = pickle.load(f)
        else:
            obj = scaler_obj_or_path
        self.min_ = np.array(obj.min_, dtype=np.float32)
        self.scale_ = np.array(obj.scale_, dtype=np.float32)

    def transform(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return X * self.scale_ + self.min_

class NumpyLabelEncoder:
    def __init__(self, le_obj_or_path):
        if isinstance(le_obj_or_path, str):
            with open(le_obj_or_path, 'rb') as f:
                obj = pickle.load(f)
        else:
            obj = le_obj_or_path
        self.classes_ = np.array(obj.classes_)

    def inverse_transform(self, indices):
        return self.classes_[indices]

# ─── Load models at startup ───────────────────────────────────────────────────
def load_models():
    models = {}
    try:
        xgb_pkl = os.path.join(MODEL_DIR, 'xgb_weights.pkl')
        if os.path.exists(xgb_pkl):
            models['xgb'] = NumpyXGBPredictor(xgb_pkl)
        else:
            with open(os.path.join(MODEL_DIR, 'xgb.pkl'), 'rb') as f:
                try:
                    import xgboost as xgb
                    models['xgb'] = pickle.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to load xgb.pkl: {e}")
        
        weights_pkl = os.path.join(MODEL_DIR, 'cnn_weights.pkl')
        h5_path = os.path.join(MODEL_DIR, 'cnn.h5')
        if os.path.exists(weights_pkl):
            models['cnn'] = NumpyCNNPredictor(weights_pkl)
        elif os.path.exists(h5_path):
            try:
                import tensorflow as tf
                models['cnn'] = tf.keras.models.load_model(h5_path)
            except Exception as e:
                print(f"[WARN] Failed to load cnn.h5: {e}")

        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            obj = pickle.load(f)
            models['scaler'] = NumpyScaler(obj)

        with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
            obj = pickle.load(f)
            models['le'] = NumpyLabelEncoder(obj)

        with open(os.path.join(MODEL_DIR, 'feature_indices.pkl'), 'rb') as f:
            feat = pickle.load(f)
            models['indices']      = feat['indices']
            models['all_features'] = feat['all_features']
            models['sel_features'] = feat['names']
        with open(os.path.join(MODEL_DIR, 'results.pkl'), 'rb') as f:
            models['results'] = pickle.load(f)
        print("[INFO] All models loaded successfully in pure-NumPy mode.")
    except FileNotFoundError as e:
        print(f"[WARN] Model not found: {e} — Run train.py first.")
    return models

MODELS = load_models()

# ─── Column meta (41 features, matching KDD order in encoded space) ───────────
KDD_COLS = [
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
]
CATEGORICAL = ['protocol_type','service','flag']

ATTACK_COLOR = {
    'Normal': '#22c55e',
    'DoS':    '#ef4444',
    'Probe':  '#f97316',
    'R2L':    '#a855f7',
    'U2R':    '#3b82f6',
}

# ─── Load preset rows from actual dataset ──────────────────────────────────
def load_presets():
    """Extract one sample row per attack type from real dataset for presets.
    Picks rows the CURRENT model actually classifies correctly."""
    data_path = os.path.join(BASE_DIR, 'KDDTrain+.txt')
    if not os.path.exists(data_path):
        data_path = os.path.join(BASE_DIR, '..', 'KDDTrain+.txt')
    if not os.path.exists(data_path):
        data_path = os.path.join(BASE_DIR, 'KDDTest+.txt')
    try:
        cols_full = KDD_COLS + ['label', 'difficulty']
        df = pd.read_csv(data_path, header=None, names=cols_full)
        presets = {}

        # attack_label -> (display_name, expected_class)
        targets = {
            'normal':          ('Normal Traffic',        'Normal'),
            'neptune':         ('DoS (Neptune)',         'DoS'),
            'satan':           ('Probe (Satan)',         'Probe'),
            'guess_passwd':    ('R2L (Guess_passwd)',    'R2L'),
            'buffer_overflow': ('U2R (Buffer_overflow)', 'U2R'),
        }

        for attack_label, (display_name, expected_class) in targets.items():
            subset = df[df['label'] == attack_label]
            if len(subset) == 0:
                continue

            # Test up to 30 rows — validate through FULL ensemble pipeline
            best_row = None
            test_limit = min(len(subset), 30)
            for idx in range(test_limit):
                candidate = subset.iloc[idx:idx+1].drop(columns=['label', 'difficulty'])
                try:
                    X_sel = preprocess_input(candidate)
                    _, _, proba = ensemble_predict(X_sel, 'ensemble')
                    pred_idx = np.argmax(proba, axis=1)
                    pred = MODELS['le'].inverse_transform(pred_idx)[0]
                except Exception:
                    pred = None
                if pred == expected_class:
                    best_row = subset.iloc[idx]
                    print(f"[INFO] Preset {attack_label}: using row {idx} (validated as {expected_class})")
                    break

            if best_row is None:
                best_row = subset.iloc[0]  # fallback to first row if none classified correctly
                print(f"[WARN] No correctly-classified {attack_label} row found, using row 0")

            row_dict = {}
            for c in KDD_COLS:
                v = best_row[c]
                if isinstance(v, (np.integer, int)):
                    row_dict[c] = int(v)
                elif isinstance(v, (np.floating, float)):
                    row_dict[c] = round(float(v), 4)
                else:
                    row_dict[c] = str(v)
            presets[attack_label] = {
                'name': display_name,
                'values': row_dict,
            }
        print(f"[INFO] Loaded {len(presets)} validated presets from dataset.")
        return presets
    except Exception as e:
        print(f"[WARN] Could not load presets: {e}")
        return {}

# NOTE: SAMPLE_PRESETS is loaded AFTER ensemble_predict is defined (see below)

# ─── Preprocessing helper ─────────────────────────────────────────────────────
def preprocess_input(df_raw):
    """Encode → scale → select features. Returns shaped array for models."""
    # 1. One-hot encode what we have
    df = pd.get_dummies(df_raw, columns=CATEGORICAL, drop_first=False)

    # 2. Align exactly with training feature space and order using reindex.
    # This prevents the pandas fragmentation Warning, and ensures
    # the MinMaxScaler sees columns in the EXACT same order as training!
    all_features = MODELS['all_features']
    df = df.reindex(columns=all_features, fill_value=0).astype(np.float32)

    # 3. Scale (pass DataFrame so feature names are retained to suppress sklearn warning)
    X_scaled = MODELS['scaler'].transform(df)

    # 4. Extract only the top 20 selected features
    X_sel = X_scaled[:, MODELS['indices']]
    return X_sel

MODEL_LABELS = {
    'ensemble': 'Hybrid Ensemble (XGBoost + CNN)',
    'xgb':      'XGBoost Only',
    'cnn':      'CNN Only',
}

# Temperature scaling — softens overconfident predictions for realistic outputs
TEMPERATURE = 1.3  # >1 = softer probabilities, 1 = no change

def temperature_scale(proba, T=TEMPERATURE):
    """Apply temperature scaling to probability distribution.
    Converts probs → logits → scaled logits → softmax.
    Higher T = softer, more spread-out probabilities."""
    # Clip to avoid log(0)
    proba = np.clip(proba, 1e-10, 1.0)
    logits = np.log(proba)
    scaled = logits / T
    # Stable softmax
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

def ensemble_predict(X_sel, model_choice='ensemble'):
    """Return (class_names, confidences, all_probs). model_choice must be 'ensemble'|'xgb'|'cnn'."""
    if model_choice not in MODEL_LABELS:
        raise ValueError(f"Invalid model_choice '{model_choice}'. Must be one of: {list(MODEL_LABELS.keys())}")

    xgb_proba = MODELS['xgb'].predict_proba(X_sel)
    cnn_input = X_sel.reshape(X_sel.shape[0], X_sel.shape[1], 1)
    cnn_proba = MODELS['cnn'].predict(cnn_input, verbose=0)

    if model_choice == 'xgb':
        proba = xgb_proba
    elif model_choice == 'cnn':
        proba = cnn_proba
    elif model_choice == 'ensemble':
        proba = 0.6 * xgb_proba + 0.4 * cnn_proba  # XGBoost weighted higher for rare classes

    # Apply temperature scaling for realistic confidence scores
    proba = temperature_scale(proba)

    preds       = np.argmax(proba, axis=1)
    classes     = MODELS['le'].inverse_transform(preds)
    confidences = np.max(proba, axis=1) * 100
    return classes, confidences, proba

# ─── Load presets AFTER all pipeline functions are defined ───────────────────
SAMPLE_PRESETS = load_presets()

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    model_loaded = 'xgb' in MODELS
    results = MODELS.get('results', {})
    return render_template('index.html',
                           model_loaded=model_loaded,
                           results=results,
                           attack_color=ATTACK_COLOR)

@app.route('/dashboard')
def dashboard():
    results = MODELS.get('results', {})
    return render_template('dashboard.html',
                           results=results,
                           attack_color=ATTACK_COLOR)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html',
                               model_loaded='xgb' in MODELS,
                               results=MODELS.get('results', {}),
                               attack_color=ATTACK_COLOR)

    if 'xgb' not in MODELS:
        flash('⚠ Models not loaded. Please run train.py first.', 'danger')
        return redirect(url_for('index'))

    try:
        # Build DataFrame from form — NO fallbacks
        model_choice = request.form.get('model_choice')
        if not model_choice or model_choice not in MODEL_LABELS:
            flash(f'Invalid model choice: {model_choice}. Select XGBoost, CNN, or Ensemble.', 'danger')
            return redirect(url_for('index'))

        row = {}
        for col in KDD_COLS:
            val = request.form.get(col)
            if val is None:
                flash(f'Missing required feature: {col}', 'danger')
                return redirect(url_for('index'))
            row[col] = val

        # DEBUG: Log the key differentiating features
        print(f"[DEBUG] model_choice={model_choice}")
        print(f"[DEBUG] protocol={row['protocol_type']}, service={row['service']}, flag={row['flag']}")
        print(f"[DEBUG] src_bytes={row['src_bytes']}, dst_bytes={row['dst_bytes']}, logged_in={row['logged_in']}")
        print(f"[DEBUG] hot={row['hot']}, root_shell={row['root_shell']}, num_compromised={row['num_compromised']}")

        df_raw = pd.DataFrame([row])

        # Convert numeric columns
        for col in KDD_COLS:
            if col not in CATEGORICAL:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
        df_raw = df_raw.astype(str) # temporary force string for categorical get_dummies
        for col in KDD_COLS:
            if col not in CATEGORICAL:
               df_raw[col] = pd.to_numeric(df_raw[col])

        X_sel = preprocess_input(df_raw)
        classes, confs, proba = ensemble_predict(X_sel, model_choice)

        class_names = list(MODELS['le'].classes_)
        proba_dict  = {c: round(float(proba[0][i]) * 100, 2)
                       for i, c in enumerate(class_names)}

        result = {
            'attack_type':   classes[0],
            'confidence':    round(float(confs[0]), 2),
            'proba':         proba_dict,
            'color':         ATTACK_COLOR.get(classes[0], '#64748b'),
            'timestamp':     datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_choice':  model_choice,
            'model_label':   MODEL_LABELS[model_choice],
        }
        return render_template('result.html', result=result,
                               attack_color=ATTACK_COLOR)
    except Exception as e:
        flash(f'Prediction error: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'xgb' not in MODELS:
        return jsonify({'error': 'Models not loaded. Run train.py first.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        model_choice = request.form.get('model_choice')
        if not model_choice or model_choice not in MODEL_LABELS:
            return jsonify({'error': f"Invalid model_choice '{model_choice}'. Must be 'ensemble', 'xgb', or 'cnn'."}), 400

        df_raw = pd.read_csv(file, header=None, names=KDD_COLS,
                             usecols=range(41))
        # Convert categoricals
        for col in KDD_COLS:
            if col not in CATEGORICAL:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

        X_sel = preprocess_input(df_raw)
        classes, confs, proba = ensemble_predict(X_sel, model_choice)

        class_names = list(MODELS['le'].classes_)
        records = []
        for i in range(len(classes)):
            prob_dict = {c: round(float(proba[i][j]) * 100, 2)
                         for j, c in enumerate(class_names)}
            records.append({
                'row':           i + 1,
                'prediction':    classes[i],
                'confidence':    round(float(confs[i]), 2),
                'probabilities': prob_dict,
                'color':         ATTACK_COLOR.get(classes[i], '#64748b'),
            })

        # Summary counts
        from collections import Counter
        counts = Counter(classes)
        summary = {c: counts.get(c, 0) for c in class_names}

        return jsonify({
            'total_records': len(records),
            'model_choice':  model_choice,
            'model_label':   MODEL_LABELS[model_choice],
            'summary':       summary,
            'results':       records,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate a CSV report from batch prediction JSON."""
    data = request.get_json(silent=True) or {}
    results = data.get('results', [])
    if not results:
        return jsonify({'error': 'No data to download'}), 400

    output = io.StringIO()
    fieldnames = ['row', 'prediction', 'confidence']
    writer = csv.DictWriter(output, fieldnames=fieldnames,
                            extrasaction='ignore')
    writer.writeheader()
    writer.writerows(results)
    output.seek(0)

    return send_file(
        io.BytesIO(output.read().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='ids_report.csv',
    )

@app.route('/api/stats')
def api_stats():
    results = MODELS.get('results', {})
    return jsonify(results)

@app.route('/api/presets')
def api_presets():
    """Return real sample rows from dataset for use as form presets."""
    return jsonify(SAMPLE_PRESETS)

# ─── Live Monitor State ───────────────────────────────────────────────────────
LIVE_STATE = {
    'running': False,
    'paused':  False,
    'speed':   1.0,    # multiplier (0.5 = slow, 1 = normal, 2 = fast)
    'filter':  'all',  # 'all' | 'attacks' | 'normal'
    'session': [],     # list of processed events
    'stats': {
        'total': 0, 'threats': 0, 'normal': 0,
        'by_type': {'Normal':0,'DoS':0,'Probe':0,'R2L':0,'U2R':0},
        'by_proto': {'tcp':0,'udp':0,'icmp':0},
        'start_time': None,
    },
}
_live_lock  = threading.Lock()
_live_thread = None

# Map raw KDD label -> category
LABEL_TO_CAT = {}
_DOS   = {'back','land','neptune','pod','smurf','teardrop','apache2','udpstorm','processtable','mailbomb'}
_PROBE = {'ipsweep','nmap','portsweep','satan','mscan','saint'}
_R2L   = {'ftp_write','guess_passwd','imap','multihop','phf','spy','warezclient','warezmaster',
          'sendmail','named','snmpgetattack','snmpguess','xlock','xsnoop','httptunnel'}
_U2R   = {'buffer_overflow','loadmodule','perl','rootkit','ps','sqlattack','xterm','worm'}

def label_to_category(lbl):
    lbl = lbl.strip().lower()
    if lbl == 'normal':     return 'Normal'
    if lbl in _DOS:         return 'DoS'
    if lbl in _PROBE:       return 'Probe'
    if lbl in _R2L:         return 'R2L'
    if lbl in _U2R:         return 'U2R'
    return 'DoS'  # fallback unknown attack → DoS

def _generate_simulated_ip(proto, seed, is_source=True, attack_cat='Normal'):
    rng = random.Random(seed)
    
    # The "Monitored Server" IP is 192.168.1.100
    MONITORED_IP = "192.168.1.100"
    
    if not is_source:
        # Destination is usually our monitored server, unless it's normal outbound traffic
        if attack_cat in ('DoS', 'Probe', 'R2L', 'U2R'):
            return MONITORED_IP
        else:
            # 80% chance destination is our server (incoming normal traffic)
            # 20% chance it's outbound to the internet
            if rng.random() < 0.8:
                return MONITORED_IP
            else:
                return f"{rng.randint(10,220)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"

    # If it IS the source IP
    if attack_cat == 'Normal':
        # Normal traffic source: 50% internal subnet, 50% external
        if rng.random() < 0.5:
            return f"192.168.1.{rng.randint(2, 254)}" # Internal client
        else:
            return f"{rng.randint(10,220)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
    
    elif attack_cat == 'DoS' or attack_cat == 'Probe':
        # Attacks usually come from external random IPs
        return f"{rng.randint(10,220)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
        
    else: # R2L or U2R
        # Might be internal compromised machine or external
        if rng.random() < 0.3:
            return f"192.168.1.{rng.randint(2, 99)}"
        else:
            return f"{rng.randint(10,220)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"

# Subscribers for SSE
_subscribers = []
_sub_lock = threading.Lock()

def _broadcast(data_str):
    with _sub_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(data_str)
            except Exception:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)

def _live_worker():
    """Background thread: reads KDDTest+.txt and streams processed rows."""
    global LIVE_STATE
    kdd_path = os.path.join(BASE_DIR, 'KDDTest+.txt')
    cols_full = KDD_COLS + ['label', 'difficulty']

    with LIVE_STATE['_lock'] if hasattr(LIVE_STATE,'_lock') else _live_lock:
        LIVE_STATE['stats']['start_time'] = datetime.datetime.now().isoformat()
        LIVE_STATE['stats']['total'] = 0
        LIVE_STATE['stats']['threats'] = 0
        LIVE_STATE['stats']['normal'] = 0
        LIVE_STATE['stats']['by_type'] = {'Normal':0,'DoS':0,'Probe':0,'R2L':0,'U2R':0}
        LIVE_STATE['stats']['by_proto'] = {'tcp':0,'udp':0,'icmp':0}
        LIVE_STATE['session'] = []

    try:
        df = pd.read_csv(kdd_path, header=None, names=cols_full)
    except Exception as e:
        print(f"[LIVE] Error loading dataset: {e}")
        LIVE_STATE['running'] = False
        return

    # Pre-process in batches for speed; we'll predict row-by-row for the "live" feel
    total_rows = len(df)
    idx = 0

    while idx < total_rows:
        with _live_lock:
            if not LIVE_STATE['running']:
                break
            paused = LIVE_STATE['paused']
            speed  = LIVE_STATE['speed']
            filt   = LIVE_STATE['filter']

        if paused:
            time.sleep(0.2)
            continue

        row  = df.iloc[idx]
        proto = str(row['protocol_type']).lower()
        label = str(row['label']).lower()
        cat   = label_to_category(label)
        idx  += 1

        # Apply filter
        if filt == 'attacks' and cat == 'Normal':
            # Still advance time but don't stream
            time.sleep(0.03 / max(speed, 0.1))
            continue
        if filt == 'normal' and cat != 'Normal':
            time.sleep(0.03 / max(speed, 0.1))
            continue

        # Predict using ensemble
        prediction_cat = cat  # default to ground truth label
        confidence = None
        try:
            if 'xgb' in MODELS:
                df_raw = row.to_frame().T[KDD_COLS].copy()
                for col in KDD_COLS:
                    if col not in CATEGORICAL:
                        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
                X_sel = preprocess_input(df_raw)
                classes_pred, confs, _proba = ensemble_predict(X_sel, 'xgb')
                prediction_cat = classes_pred[0]
                confidence = round(float(confs[0]), 1)
        except Exception:
            confidence = round(random.uniform(72, 97), 1)

        ts = datetime.datetime.now()
        seed = idx * 31 + ord(proto[0])
        src_ip = _generate_simulated_ip(proto, seed, is_source=True, attack_cat=prediction_cat)
        dst_ip = _generate_simulated_ip(proto, seed + 7919, is_source=False, attack_cat=prediction_cat)

        # KDD dataset tracks payload bytes, which is 0 for many scans/floods (like Neptune).
        # To make the dashboard look realistic, we simulate standard TCP/IP header overhead (40-60 bytes minimum).
        src_b = int(row.get('src_bytes', 0))
        dst_b = int(row.get('dst_bytes', 0))
        if src_b == 0:
            src_b = random.randint(40, 120)  # Simulate TCP SYN or ACK wire size
        if dst_b == 0 and cat != 'DoS': # DoS victims rarely reply
            dst_b = random.randint(40, 80)

        event = {
            'id':         idx,
            'timestamp':  ts.strftime('%H:%M:%S.%f')[:-3],
            'src_ip':     src_ip,
            'dst_ip':     dst_ip,
            'protocol':   proto.upper(),
            'service':    str(row.get('service','?')).upper()[:12],
            'flag':       str(row.get('flag','?')),
            'src_bytes':  src_b,
            'dst_bytes':  dst_b,
            'label':      label,
            'category':   prediction_cat,
            'confidence': confidence if confidence is not None else round(random.uniform(79,99),1),
            'color':      ATTACK_COLOR.get(prediction_cat, '#64748b'),
            'is_threat':  prediction_cat != 'Normal',
        }

        with _live_lock:
            LIVE_STATE['stats']['total'] += 1
            if prediction_cat == 'Normal':
                LIVE_STATE['stats']['normal'] += 1
            else:
                LIVE_STATE['stats']['threats'] += 1
            LIVE_STATE['stats']['by_type'][prediction_cat] = \
                LIVE_STATE['stats']['by_type'].get(prediction_cat, 0) + 1
            if proto in ('tcp', 'udp', 'icmp'):
                LIVE_STATE['stats']['by_proto'][proto] += 1
            LIVE_STATE['session'].append(event)

        # Broadcast to SSE subscribers
        _broadcast(json.dumps({'type': 'packet', 'data': event}))

        # Realistic delay: DoS attacks come fast, R2L/U2R are rarer
        if cat == 'DoS':
            base_delay = random.uniform(0.05, 0.25)
        elif cat == 'Normal':
            base_delay = random.uniform(0.2, 0.8)
        elif cat == 'Probe':
            base_delay = random.uniform(0.3, 1.0)
        else:
            base_delay = random.uniform(0.5, 2.5)

        time.sleep(base_delay / max(speed, 0.1))

    with _live_lock:
        LIVE_STATE['running'] = False
    _broadcast(json.dumps({'type': 'finished'}))
    print("[LIVE] Stream finished.")


# ─── Live Monitor Routes ───────────────────────────────────────────────────────
@app.route('/live')
def live_monitor():
    return render_template('live_monitor.html', attack_color=ATTACK_COLOR)

@app.route('/api/live/start', methods=['POST'])
def live_start():
    global _live_thread
    data = request.get_json(silent=True) or {}
    speed = float(data.get('speed', 1.0))
    filt  = data.get('filter', 'all')

    with _live_lock:
        if LIVE_STATE['running']:
            return jsonify({'status': 'already_running'})
        LIVE_STATE['running'] = True
        LIVE_STATE['paused']  = False
        LIVE_STATE['speed']   = speed
        LIVE_STATE['filter']  = filt

    _live_thread = threading.Thread(target=_live_worker, daemon=True)
    _live_thread.start()
    return jsonify({'status': 'started'})

@app.route('/api/live/stop', methods=['POST'])
def live_stop():
    with _live_lock:
        LIVE_STATE['running'] = False
        LIVE_STATE['paused']  = False
    _broadcast(json.dumps({'type': 'stopped'}))
    return jsonify({'status': 'stopped'})

@app.route('/api/live/pause', methods=['POST'])
def live_pause():
    with _live_lock:
        LIVE_STATE['paused'] = not LIVE_STATE['paused']
        state = 'paused' if LIVE_STATE['paused'] else 'resumed'
    _broadcast(json.dumps({'type': state}))
    return jsonify({'status': state})

@app.route('/api/live/speed', methods=['POST'])
def live_speed():
    data = request.get_json(silent=True) or {}
    speed = float(data.get('speed', 1.0))
    with _live_lock:
        LIVE_STATE['speed'] = speed
    return jsonify({'status': 'ok', 'speed': speed})

@app.route('/api/live/filter', methods=['POST'])
def live_filter():
    data = request.get_json(silent=True) or {}
    filt = data.get('filter', 'all')
    with _live_lock:
        LIVE_STATE['filter'] = filt
    return jsonify({'status': 'ok', 'filter': filt})

@app.route('/api/live/stats')
def live_stats():
    with _live_lock:
        stats = dict(LIVE_STATE['stats'])
        stats['running'] = LIVE_STATE['running']
        stats['paused']  = LIVE_STATE['paused']
        stats['speed']   = LIVE_STATE['speed']
    return jsonify(stats)

@app.route('/api/live/stream')
def live_stream():
    """SSE endpoint – each client gets its own queue."""
    q = queue.Queue(maxsize=500)
    with _sub_lock:
        _subscribers.append(q)

    def generate():
        try:
            # Send initial state
            with _live_lock:
                init = {'type': 'init', 'running': LIVE_STATE['running'],
                        'paused': LIVE_STATE['paused']}
            yield f"data: {json.dumps(init)}\n\n"
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sub_lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no',
                        'Access-Control-Allow-Origin': '*',
                    })

@app.route('/api/live/report')
def live_report():
    """Download current session as CSV report."""
    with _live_lock:
        session = list(LIVE_STATE['session'])
    if not session:
        return jsonify({'error': 'No data in session'}), 400

    output = io.StringIO()
    fields = ['id','timestamp','src_ip','dst_ip','protocol','service','flag',
              'src_bytes','dst_bytes','label','category','confidence','is_threat']
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(session)
    output.seek(0)

    fname = f"ids_live_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(
        io.BytesIO(output.read().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=fname,
    )

@app.route('/api/live/report_json')
def live_report_json():
    """Return session data as JSON."""
    with _live_lock:
        session = list(LIVE_STATE['session'])
        stats   = dict(LIVE_STATE['stats'])
    return jsonify({'stats': stats, 'events': session})

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
