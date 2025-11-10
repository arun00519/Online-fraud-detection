from flask import Flask, render_template, request, jsonify
import os, joblib, json, pandas as pd

app = Flask(__name__)

# -------------------------------
# Load model and its info
# -------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
model_features = model_data['features']
threshold = model_data.get('threshold', 0.35)

# -------------------------------
# Rule-based detection
# -------------------------------
def rule_flag(data):
    amount = float(data.get('amount', 0))
    old_org = float(data.get('oldbalanceOrg', 0))
    new_org = float(data.get('newbalanceOrig', 0))
    old_dest = float(data.get('oldbalanceDest', 0))
    new_dest = float(data.get('newbalanceDest', 0))
    flags = []

    if old_org == 0 and amount > 0:
        flags.append('Sender had zero balance but sent money')
    if (old_org - new_org) != amount and amount > 0:
        flags.append('Sender balance mismatch')
    if (new_dest - old_dest) == amount and old_org == 0:
        flags.append('Receiver credited full amount while sender unchanged')
    if amount > 5000:
        flags.append('Large transaction amount')

    return flags


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html', model_available=True)


@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'amount': request.form.get('amount', 0),
        'oldbalanceOrg': request.form.get('oldbalanceOrg', 0),
        'newbalanceOrig': request.form.get('newbalanceOrig', 0),
        'oldbalanceDest': request.form.get('oldbalanceDest', 0),
        'newbalanceDest': request.form.get('newbalanceDest', 0),
        'type': request.form.get('type', 'TRANSFER')
    }

    # Validate inputs
    try:
        amount = float(data['amount'])
        old_org = float(data['oldbalanceOrg'])
        new_org = float(data['newbalanceOrig'])
        old_dest = float(data['oldbalanceDest'])
        new_dest = float(data['newbalanceDest'])
    except ValueError:
        return render_template('index.html',
                               error='Please enter valid numeric values',
                               model_available=True)

    # Encode transaction type
    type_cash_out = 1 if data['type'] == 'CASH_OUT' else 0
    type_transfer = 1 if data['type'] == 'TRANSFER' else 0

    # Feature engineering (same as training)
    org_change = old_org - new_org
    dest_change = new_dest - old_dest
    amount_ratio = amount / (old_org + 1)
    is_sender_zero = int(old_org == 0)
    is_full_credit = int(dest_change == amount)

    # Create feature vector
    input_dict = {
        'step': 1,
        'amount': amount,
        'oldbalanceOrg': old_org,
        'newbalanceOrig': new_org,
        'oldbalanceDest': old_dest,
        'newbalanceDest': new_dest,
        'type_CASH_OUT': type_cash_out,
        'type_TRANSFER': type_transfer,
        'org_change': org_change,
        'dest_change': dest_change,
        'amount_ratio': amount_ratio,
        'is_sender_zero': is_sender_zero,
        'is_full_credit': is_full_credit
    }

    X = pd.DataFrame([[input_dict.get(col, 0) for col in model_features]],
                     columns=model_features)

    # Model prediction
    prob = model.predict_proba(X)[0][1]
    is_model_fraud = prob >= threshold

    # Rule-based check
    flags = rule_flag(data)
    is_rule_fraud = len(flags) > 0

    # Final decision: if ANY says fraud → fraud
    if is_model_fraud or is_rule_fraud:
        result = '🚨 Fraudulent Transaction'
    else:
        result = '✅ Legitimate Transaction'

    return render_template('index.html',
                           result=result,
                           probability=round(float(prob), 3),
                           flags=flags,
                           model_available=True)


if __name__ == '__main__':
    app.run(debug=True)
