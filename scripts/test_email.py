import smtplib
import os

sender = "debapriya14jis@gmail.com"  # use the same Gmail you used to generate the app password
receiver = "debrachoubey14jis@gmail.com"  # or any email you want to receive the test
password = "vnmissfyssvtarjc"

try:
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        print("✅ Gmail SMTP login successful")
except Exception as e:
    print("❌ Login failed:", e)
