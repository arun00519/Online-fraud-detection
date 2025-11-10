
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import json, numpy as np

df = pd.read_csv("dataset/sample_fraud_data.csv")
df = df[df['type'].isin(['TRANSFER','CASH_OUT'])]
df = pd.get_dummies(df, columns=['type'], drop_first=False)
for c in ['nameOrig','nameDest','isFlaggedFraud']:
    if c in df.columns:
        df = df.drop(columns=c)
df['org_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['dest_change'] = df['newbalanceDest'] - df['oldbalanceDest']
df['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
df['is_sender_zero'] = (df['oldbalanceOrg']==0).astype(int)
df['is_full_credit'] = (df['dest_change']==df['amount']).astype(int)

X = df.drop('isFraud', axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

sm = SMOTEENN(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

clf = RandomForestClassifier(class_weight='balanced_subsample', n_estimators=300, random_state=42)
clf.fit(X_res, y_res)

y_scores = clf.predict_proba(X_test)[:,1]
prec, rec, thr = precision_recall_curve(y_test, y_scores)
beta=2.0
f2 = (1+beta**2)*(prec*rec)/((beta**2)*prec + rec)
import numpy as np
best_idx = int(np.nanargmax(f2))
best_threshold = float(thr[best_idx]) if best_idx < len(thr) else 0.5

joblib.dump({'model':clf, 'features': list(X.columns), 'threshold': best_threshold}, 'model.pkl')
with open('model_info.json','w') as f:
    json.dump({'features': list(X.columns), 'threshold': best_threshold}, f, indent=4)

print("Trained and saved model with threshold:", best_threshold)
print("Classification report:")
y_pred = (y_scores >= best_threshold).astype(int)
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))