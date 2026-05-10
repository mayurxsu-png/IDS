import pickle, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Load models
with open('model/xgb.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
cnn_model = tf.keras.models.load_model('model/cnn.h5')
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('model/feature_indices.pkl', 'rb') as f:
    feat = pickle.load(f)
    indices = feat['indices']
    all_features = feat['all_features']

# Load test data
import pandas as pd
COLUMNS = ['duration','protocol_type','service','flag','src_bytes','dst_bytes',
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
    'label','difficulty']

ATTACK_MAP = {
    'normal': 'Normal','back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS','worm':'DoS','mailbomb':'DoS',
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe','mscan':'Probe','saint':'Probe',
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L','phf':'R2L','spy':'R2L',
    'warezclient':'R2L','warezmaster':'R2L','xlock':'R2L','xsnoop':'R2L','snmpguess':'R2L',
    'snmpgetattack':'R2L','httptunnel':'R2L','sendmail':'R2L','named':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R','ps':'U2R','xterm':'U2R','sqlattack':'U2R',
}

print('Loading test data...')
df = pd.read_csv('../KDDTrain+.txt', header=None, names=COLUMNS)
df['label'] = df['label'].str.strip()
df['attack_cat'] = df['label'].map(ATTACK_MAP).fillna('Normal')
df.drop(columns=['label','difficulty'], inplace=True)

# Preprocess
df_encoded = pd.get_dummies(df, columns=['protocol_type','service','flag'], drop_first=False)
df_aligned = df_encoded.reindex(columns=all_features, fill_value=0).astype(np.float32)
X_scaled = scaler.transform(df_aligned)
X_sel = X_scaled[:, indices]

# Encode labels
y_true = le.transform(df['attack_cat'])

print('Making predictions...')
# XGBoost predictions
xgb_proba = xgb_model.predict_proba(X_sel)
xgb_preds = xgb_model.predict(X_sel)

# CNN predictions  
X_cnn = X_sel.reshape(X_sel.shape[0], X_sel.shape[1], 1)
cnn_proba = cnn_model.predict(X_cnn, verbose=0)
cnn_preds = np.argmax(cnn_proba, axis=1)

# Ensemble predictions
ensemble_proba = 0.6 * xgb_proba + 0.4 * cnn_proba
ensemble_preds = np.argmax(ensemble_proba, axis=1)

print('\n=== ENSEMBLE MODEL METRICS ===')
class_names = le.classes_
report = classification_report(y_true, ensemble_preds, target_names=class_names, output_dict=True)
cm = confusion_matrix(y_true, ensemble_preds)

# Calculate metrics for each class
print('\nPer-Class Metrics:')
for i, class_name in enumerate(class_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - (tp + fp + fn)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f'{class_name:10s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, FPR={fpr:.3f}')

print('\nOverall Metrics:')
print(f'Accuracy: {(y_true == ensemble_preds).mean():.3f}')
print(f'Macro Precision: {report["macro avg"]["precision"]:.3f}')
print(f'Macro Recall: {report["macro avg"]["recall"]:.3f}')
print(f'Macro F1: {report["macro avg"]["f1-score"]:.3f}')
print(f'Weighted Precision: {report["weighted avg"]["precision"]:.3f}')
print(f'Weighted Recall: {report["weighted avg"]["recall"]:.3f}')
print(f'Weighted F1: {report["weighted avg"]["f1-score"]:.3f}')

print('\n=== XGBOOST MODEL METRICS ===')
report_xgb = classification_report(y_true, xgb_preds, target_names=class_names, output_dict=True)
print(f'Accuracy: {(y_true == xgb_preds).mean():.3f}')
print(f'Macro F1: {report_xgb["macro avg"]["f1-score"]:.3f}')
print(f'Weighted F1: {report_xgb["weighted avg"]["f1-score"]:.3f}')

print('\n=== CNN MODEL METRICS ===')
report_cnn = classification_report(y_true, cnn_preds, target_names=class_names, output_dict=True)
print(f'Accuracy: {(y_true == cnn_preds).mean():.3f}')
print(f'Macro F1: {report_cnn["macro avg"]["f1-score"]:.3f}')
print(f'Weighted F1: {report_cnn["weighted avg"]["f1-score"]:.3f}')

print('\nConfusion Matrix (Ensemble):')
print(cm)
