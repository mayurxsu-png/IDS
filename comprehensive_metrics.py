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

class_names = le.classes_

def calculate_detailed_metrics(y_true, y_preds, model_name):
    """Calculate precision, recall, F1, and FPR for each class"""
    print(f'\n{"="*60}')
    print(f'{model_name} MODEL - DETAILED METRICS')
    print(f'{"="*60}')
    
    cm = confusion_matrix(y_true, y_preds)
    report = classification_report(y_true, y_preds, target_names=class_names, output_dict=True)
    
    print(f'\n📊 Per-Class Performance:')
    print(f'{"Attack Type":<12} {"Precision":<10} {"Recall":<10} {"F1-Score":<10} {"FPR":<10} {"Support":<8}')
    print(f'{"-"*70}')
    
    metrics_data = []
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, :].sum()
        
        print(f'{class_name:<12} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {fpr:<10.3f} {support:<8}')
        metrics_data.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'support': support
        })
    
    print(f'\n📈 Overall Performance:')
    accuracy = (y_true == y_preds).mean()
    print(f'Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)')
    print(f'Macro Precision: {report["macro avg"]["precision"]:.3f}')
    print(f'Macro Recall: {report["macro avg"]["recall"]:.3f}')
    print(f'Macro F1-Score: {report["macro avg"]["f1-score"]:.3f}')
    print(f'Weighted Precision: {report["weighted avg"]["precision"]:.3f}')
    print(f'Weighted Recall: {report["weighted avg"]["recall"]:.3f}')
    print(f'Weighted F1-Score: {report["weighted avg"]["f1-score"]:.3f}')
    
    # Calculate average FPR
    avg_fpr = sum(m['fpr'] for m in metrics_data) / len(metrics_data)
    print(f'Average False Positive Rate: {avg_fpr:.3f} ({avg_fpr*100:.1f}%)')
    
    print(f'\n🔍 Confusion Matrix:')
    print(f'{"Predicted":>12}', end='')
    for name in class_names:
        print(f'{name:>8}', end='')
    print(f' {"Total":>8}')
    
    for i, true_class in enumerate(class_names):
        print(f'{true_class:>12}', end='')
        for j in range(len(class_names)):
            print(f'{cm[i,j]:>8}', end='')
        print(f' {cm[i,:].sum():>8}')
    
    print(f'{"Total":>12}', end='')
    for j in range(len(class_names)):
        print(f'{cm[:,j].sum():>8}', end='')
    print(f' {cm.sum():>8}')
    
    return metrics_data, accuracy, avg_fpr

# Calculate metrics for all models
xgb_metrics, xgb_acc, xgb_fpr = calculate_detailed_metrics(y_true, xgb_preds, "XGBOOST")
cnn_metrics, cnn_acc, cnn_fpr = calculate_detailed_metrics(y_true, cnn_preds, "CNN")
ensemble_metrics, ensemble_acc, ensemble_fpr = calculate_detailed_metrics(y_true, ensemble_preds, "ENSEMBLE")

print(f'\n{"="*80}')
print(f'🏆 MODEL COMPARISON SUMMARY')
print(f'{"="*80}')

print(f'\n📊 Overall Performance Comparison:')
print(f'{"Model":<12} {"Accuracy":<10} {"Avg FPR":<10} {"Macro F1":<10} {"Weighted F1":<12}')
print(f'{"-"*60}')
print(f'{"XGBoost":<12} {xgb_acc:<10.3f} {xgb_fpr:<10.3f} {xgb_metrics[0]["f1"]+xgb_metrics[1]["f1"]+xgb_metrics[2]["f1"]+xgb_metrics[3]["f1"]+xgb_metrics[4]["f1"]:<10.3f} {"N/A":<12}')
print(f'{"CNN":<12} {cnn_acc:<10.3f} {cnn_fpr:<10.3f} {"N/A":<10} {"N/A":<12}')
print(f'{"Ensemble":<12} {ensemble_acc:<10.3f} {ensemble_fpr:<10.3f} {"N/A":<10} {"N/A":<12}')

print(f'\n🎯 Best Model by Metric:')
print(f'Highest Accuracy: {"Ensemble" if ensemble_acc >= max(xgb_acc, cnn_acc) else "XGBoost" if xgb_acc >= cnn_acc else "CNN"} ({max(ensemble_acc, xgb_acc, cnn_acc):.3f})')
print(f'Lowest FPR: {"Ensemble" if ensemble_fpr <= min(xgb_fpr, cnn_fpr) else "XGBoost" if xgb_fpr <= cnn_fpr else "CNN"} ({min(ensemble_fpr, xgb_fpr, cnn_fpr):.3f})')

print(f'\n📈 Per-Class Best Performers:')
for i, class_name in enumerate(class_names):
    xgb_prec = xgb_metrics[i]['precision']
    cnn_prec = cnn_metrics[i]['precision'] if i < len(cnn_metrics) else 0
    ensemble_prec = ensemble_metrics[i]['precision']
    
    best_prec = max(xgb_prec, cnn_prec, ensemble_prec)
    if best_prec == xgb_prec:
        best_model = "XGBoost"
    elif best_prec == cnn_prec:
        best_model = "CNN"
    else:
        best_model = "Ensemble"
    
    print(f'{class_name:<8}: {best_model:<10} (Precision: {best_prec:.3f})')

print(f'\n{"="*80}')
print(f'✅ ANALYSIS COMPLETE')
print(f'{"="*80}')
