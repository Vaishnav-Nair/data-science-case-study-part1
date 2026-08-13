import os
import urllib.request

os.makedirs('data', exist_ok=True)

urls = {
    'data/telco_churn.csv': 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv',
    'data/online_retail.csv': 'https://raw.githubusercontent.com/scheckley/online-retail/master/data.csv'
}

for path, url in urls.items():
    print(f"Downloading {path}...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")

print("Done. Run churn_analysis.py and retail_analysis.py next.")
