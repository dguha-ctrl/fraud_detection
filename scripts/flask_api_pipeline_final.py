from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pyodbc
import smtplib
from email.mime.text import MIMEText
import os

app = Flask(__name__)

# Load trained XGBoost pipeline
pipeline = joblib.load("C:/Users/Debrachoubey/PycharmProjects/pythonProject/fraud_detection_pipeline.pkl")
print(" Loaded XGBoost pipeline")

# SQL Server connection string
conn_str = (
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=DESKTOP-EPOVRA0\\SQLEXPRESS;'
    'DATABASE=credit_card;'
    'Trusted_Connection=yes;'
)

# Email Alert Function
def send_email_alert(transaction_id):
    sender = "debapriya14jis@gmail.com"
    receiver = "debrachoubey14jis@gmail.com"
    subject = f"Fraud Alert: Transaction {transaction_id}"
    body = f" Suspicious Transaction Detected!\nTransaction ID: {transaction_id}"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender, "vnmissfyssvtarjc")  # Replace with app password
        server.sendmail(sender, receiver, msg.as_string())
        print(" Email alert sent!")

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("🚀 /predict called")
        json_data = request.json
        df_raw = pd.DataFrame([json_data])

        # Run prediction using the pipeline
        y_proba = pipeline.predict_proba(df_raw)[:, 1]
        y_pred = int(y_proba[0] >= 0.5)

        # Update SQL Server
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE dbo.credit_card_transact SET is_fraud = ? WHERE transaction_id = ?",
                (y_pred, json_data['transaction_id'])
            )

            cursor.execute(
                "SELECT COUNT(*) FROM dbo.fraud_logs WHERE transaction_id = ?",
                (json_data['transaction_id'],)
            )
            existing_count = cursor.fetchone()[0]

            if existing_count == 0:
             cursor.execute(
                """
                INSERT INTO dbo.fraud_logs (transaction_id, merchant, country, amount, fraud_score, is_fraud)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    json_data['transaction_id'],
                    json_data.get('merchant'),
                    json_data.get('country'),
                    json_data.get('amount', 0.0),
                    float(y_proba[0]),
                    y_pred
                )
            )
             print(" Inserted fraud log:", json_data['transaction_id'])
            else:
             print(" Already logged, skipping:", json_data['transaction_id'])

            conn.commit()

        # Send alert
        if y_pred == 1:
            send_email_alert(json_data['transaction_id'])

        return jsonify({
            "transaction_id": json_data['transaction_id'],
            "prediction": y_pred,
            "xgb_score": float(y_proba[0])
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
