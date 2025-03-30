import requests
import pyodbc

url = "http://127.0.0.1:5000/predict"
data = {
  "transaction_id": "TX_08be31f1",
  "customer_id": "CUST_48086",
  "card_number": "5.35219980258509E+15",
  "merchant_category": "Grocery",
  "merchant_type": "online",
  "merchant": "FreshDirect",
  "amount": 396.170013427734,
  "currency": "EUR",
  "country": "France",
  "city": "Unknown City",
  "city_size": "medium",
  "card_type": "Premium Debit",
  "card_present": True,
  "device": "NFC Payment",
  "channel": "pos",
  "device_fingerprint": "3ff518a9de0ebe4ba43a25d90fd96481",
  "ip_address": "128.113.54.202",
  "distance_from_home": 1,
  "transaction_hour": 0,
  "weekend_transaction": False,
  "velocity_last_hour": """{
    "num_transactions": 251,
    "total_amount": 10337517.795460586,
    "unique_merchants": 92,
    "unique_countries": 12,
    "max_single_amount": 5372317.654262363
  }"""
}
response = requests.post(url, json=data)
print(response)
print(response.json())





