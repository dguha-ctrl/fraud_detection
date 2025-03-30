import requests
import pyodbc

# Connect to SQL Server
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=DESKTOP-EPOVRA0\\SQLEXPRESS;'
    'DATABASE=credit_card;'
    'Trusted_Connection=yes;'
    'connection timeout=120;'
)
cursor = conn.cursor()

# Fetch a transaction where is_fraud is NULL
cursor.execute("""
    SELECT TOP 1 
        transaction_id,
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
        is_fraud
    FROM dbo.credit_card_transact 
    WHERE date = '2024-09-30' AND is_fraud IS NULL;
""")
row = cursor.fetchone()

if row:
    (
        transaction_id, customer_id, card_number, merchant_category, merchant_type,
        merchant, amount, currency, country, city, city_size, card_type,
        card_present, device, channel, device_fingerprint, ip_address,
        distance_from_home, transaction_hour, weekend_transaction,
        velocity_last_hour, is_fraud
    ) = row

    print(f"🚀 Testing transaction: {transaction_id}")

    api_url = "http://localhost:5000/predict"
    api_payload = {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "card_number": float(card_number),
        "merchant_category": merchant_category,
        "merchant_type": merchant_type,
        "merchant": merchant,
        "amount": float(amount),
        "currency": currency,
        "country": country,
        "city": city,
        "city_size": city_size,
        "card_type": card_type,
        "card_present": bool(card_present),
        "device": device,
        "channel": channel,
        "device_fingerprint": device_fingerprint,
        "ip_address": ip_address,
        "distance_from_home": float(distance_from_home),
        "transaction_hour": int(transaction_hour),
        "weekend_transaction": bool(weekend_transaction),
        "velocity_last_hour": velocity_last_hour,
        #"date": str(date)
    }

    response = requests.post(api_url, json=api_payload)

    if response.status_code == 200:
        result = response.json()
        print(f"\n Prediction: {'FRAUD' if result['prediction'] == 1 else 'NOT FRAUD'}")
        print(f"XGBoost Score: {result['xgb_score']:.4f}")
    else:
        print(" API Error:", response.text)
else:
    print(" No transaction found with is_fraud = NULL.")

conn.close()