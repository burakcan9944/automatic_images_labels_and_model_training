import requests

url = "http://ip:port/kill_training"

try:
    response = requests.post(url)
    if response.status_code == 200:
        print("Training process killed successfully.")
    else:
        print(f"Failed to kill training process: {response.json().get('error')}")
except requests.RequestException as e:
    print(f"Error sending kill request: {e}")
