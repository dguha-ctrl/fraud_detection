from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pyodbc
import smtplib
from email.mime.text import MIMEText
import os
from datetime import datetime

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
     'connection timeout=120;'
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
@app.route("/")
def home():
    return "Fraud Detection App is running"
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(" /predict called")
        json_data = request.get_json()
        df_raw = pd.DataFrame([json_data])
        print('df_raw',df_raw)


        # Run prediction using the pipeline
        y_proba = pipeline.predict_proba(df_raw)[:, 1]
        y_pred = int(y_proba[0] >= 0.5)
        # response
        print ('ypred',y_pred)
        response = {
            "is_Fraud": "Fraud" if y_pred == 1 else "No Fraud"
        }



        # Update SQL Server
        with pyodbc.connect(conn_str) as conn:
         cursor = conn.cursor()
         current_time = datetime.now()

         cursor.execute(
                """
                INSERT INTO dbo.credit_card_newtble (transaction_id,
        customer_id,
        card_number,
        merchant_category,
        merchant_type,
        merchant,
        amount,
        currency,
        country,
        city,
        city_size,
        card_type,
        card_present,
        device,
        channel,
        device_fingerprint,
        ip_address,
        distance_from_home,
        transaction_hour,
        weekend_transaction,
        velocity_last_hour,
        is_fraud,
        Date)
                VALUES (?, ?, ?, ?, ?, ?, ? ,?, ?, ?, ?, ?, ?, ? ,?, ?, ?, ?, ?, ?, ? ,?, ?)
                """,
                (

                    json_data['transaction_id'],json_data['customer_id'],json_data['card_number'],json_data['merchant_category'],json_data['merchant_type'],json_data['merchant'],json_data['amount'],json_data['currency'],json_data['country'],json_data['city'],json_data['city_size'],json_data['card_type'],json_data['card_present'],json_data['device'],json_data['channel'],json_data['device_fingerprint'],json_data['ip_address'],json_data['distance_from_home'],json_data['transaction_hour'],json_data['weekend_transaction'],json_data['velocity_last_hour'],y_pred,current_time
                )
            )
        print(" Inserted records :", json_data['transaction_id'])
            #else:
             #print(" Already logged, skipping:", json_data['transaction_id'])

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
        print(" ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
