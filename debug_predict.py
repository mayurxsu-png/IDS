"""Find a buffer_overflow row the CURRENT retrained model classifies as U2R."""
import pickle, numpy as np, pandas as pd, json

with open('model/xgb.pkl','rb') as f: xgb = pickle.load(f)
with open('model/scaler.pkl','rb') as f: scaler = pickle.load(f)
with open('model/label_encoder.pkl','rb') as f: le = pickle.load(f)
with open('model/feature_indices.pkl','rb') as f: feat = pickle.load(f)

indices = feat['indices']
all_feats = feat['all_features']

COLS = ['duration','protocol_type','service','flag','src_bytes','dst_bytes',
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
CAT = ['protocol_type','service','flag']

df = pd.read_csv('../KDDTrain+.txt', header=None, names=COLS)

# Test all U2R-type attacks
for attack in ['buffer_overflow', 'rootkit', 'loadmodule', 'perl']:
    subset = df[df['label']==attack]
    if len(subset) == 0:
        continue
    feats = subset.drop(columns=['label','difficulty'])
    enc = pd.get_dummies(feats, columns=CAT, drop_first=False)
    ali = enc.reindex(columns=all_feats, fill_value=0).astype(np.float32)
    sc = scaler.transform(ali)
    sel = sc[:, indices]
    preds = le.inverse_transform(xgb.predict(sel))
    correct = sum(1 for p in preds if p == 'U2R')
    print(f"{attack}: {correct}/{len(preds)} correctly predicted as U2R")
    
    # Find the first correctly classified row
    for i, p in enumerate(preds):
        if p == 'U2R':
            row = subset.iloc[i]
            row_dict = {}
            for c in COLS[:41]:
                v = row[c]
                if isinstance(v, (np.integer, int)):
                    row_dict[c] = int(v)
                elif isinstance(v, (np.floating, float)):
                    row_dict[c] = round(float(v), 4)
                else:
                    row_dict[c] = str(v)
            print(f"  GOOD ROW (idx {i}): {json.dumps(row_dict)}")
            break
