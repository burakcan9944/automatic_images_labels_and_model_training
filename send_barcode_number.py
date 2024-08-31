import requests

def send_barcodes(barcodes):
    url = 'http://ip:port/barcode'
    barcode_data = {'barcodes': barcodes}
    try:
        print(f"Sending data to {url}: {barcode_data}")
        response = requests.post(url, json=barcode_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

barcodes = ['5000']

result = send_barcodes(barcodes)
#print(f"Barcodes: {barcodes} - Result: {result}")
