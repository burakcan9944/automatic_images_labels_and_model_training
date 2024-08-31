import requests

url = "http://ip:port/start_training"

try:
    response = requests.post(url)
    if response.status_code == 200:
        print("Training process started successfully.")
    else:
        print(f"Failed to start training process: {response.json().get('error')}")
except requests.RequestException as e:
    print(f"Error sending start request: {e}")
