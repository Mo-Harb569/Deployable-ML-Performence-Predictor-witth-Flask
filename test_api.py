import requests
import json

url = 'http://127.0.0.1:5000/predict_performance'

student_data = {
    "features": [7, 90, 8, 5, 1]
}

print(f"--- Sending student data: {student_data} ---")

try:
    response = requests.post(url, json=student_data)
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Model Response:")
        print(f"Predicted Performance Index is: {result['predicted_performance_index']}")
    else:
        print("❌ Error occurred:", response.text)

except Exception as e:
    print("Ensure app2.py is running in the background!")
    print(e)