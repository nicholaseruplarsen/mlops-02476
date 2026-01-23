"""Script to trigger alert by sending many requests to the API."""
import requests
import time

URL = "https://arxiv-api-286633827075.europe-west1.run.app/predict"

# Send valid requests to generate traffic
for i in range(50):
    try:
        r = requests.post(
            URL,
            json={"title": f"Test paper {i}", "abstract": "Testing the alerting system."},
            timeout=30,
        )
        print(f"Request {i+1}: {r.status_code}")
    except Exception as e:
        print(f"Request {i+1}: Error - {e}")
    time.sleep(0.5)

print("\nDone! Check Cloud Monitoring for alerts.")
