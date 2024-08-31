import eventlet
import eventlet.wsgi
from flask import Flask, send_file, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import requests
import subprocess
import os
import yaml
import training





app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode='threading',cors_allowed_origins="*")

weights_path = ""
BARCODES_FILE = 'barcodes.json'
a = True
training_status = False
training_process = None
image_labels = None
productname = None
productprice = None
current_epoch = 0

def load_barcodes():
    try:
        with open(BARCODES_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_barcodes(barcodes):
    with open(BARCODES_FILE, 'w') as f:
        json.dump(barcodes, f)

def save_barcode_to_file_write(barcodes):
    with open('barcodes_write.txt', 'w') as f:
        for barcode in barcodes:
            f.write(barcode + '\n')

def save_barcode_to_file_append(barcodes):
    with open('barcodes_append.txt', 'a') as f:
        for barcode in barcodes:
            f.write(barcode + '\n')



def get_classes_info(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    class_names = data['names']
    num_classes = len(data['names'])
    return num_classes, class_names

def save_barcode_info_to_file(update_barcodes):
    with open('barcode_info.txt', 'a') as f:
        for barcode_info in update_barcodes:
            f.write(json.dumps(barcode_info) + '\n')

def is_barcode_saved(barcode):
    try:

        if isinstance(barcode, list):
            barcodeeee = ''.join(barcode)
        with open('barcodes_append.txt', 'r') as f:
            saved_barcodes = f.read().splitlines()
            #print(f"Checking barcode: {barcode}, Saved barcodes: {saved_barcodes}")  # Debug çıktısı
            return barcodeeee in saved_barcodes
    except FileNotFoundError:
        return False

@app.route('/barcode', methods=['POST'])
def receive_barcodes():
    global a
    data = request.get_json()
    if 'barcodes' not in data:
        return jsonify({"error": "Barcode list is required"}), 400

    barcodes = data['barcodes']
    if not isinstance(barcodes, list):
        return jsonify({"error": "Barcodes must be a list"}), 400

    isSaved=is_barcode_saved(barcodes)
    existing_barcodes = load_barcodes()
    update_barcodes = []
    for item in barcodes:
        barcode_info = {
            'barcode': item,
            'productname': 'null',
            'productprice': 'null',
            'isSaved': isSaved
        }
        existing_barcodes.append(barcode_info)
        update_barcodes.append(barcode_info)


    #socketio.emit('training_status', {'update_barcodes': update_barcodes})

    if a:
        a = False
        save_barcodes(existing_barcodes)

    save_barcode_to_file_write(barcodes)
    save_barcode_info_to_file(update_barcodes)
    if isSaved == False:

        save_barcode_to_file_append(barcodes)
        image_labels_create()


    response_data = ', '.join([json.dumps(barcode) for barcode in update_barcodes])
    print(f"Received barcodes: {response_data}")
    return jsonify({"barcodes": response_data}), 200

@app.route('/image_labels_create', methods=['POST'])
def image_labels_create():
    global image_labels
    try:
        image_labels = subprocess.Popen(['python', 'images_labels.py'])
        return jsonify({"message": "image_labels_create started"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_process, training_status
    try:
        training_process = subprocess.Popen(['python', 'training.py'])
        training_status = True
        return jsonify({"message": "Training started"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/classes_info', methods=['GET'])
def classes_info():
    yaml_file_path = 'barcodes.yaml'
    try:
        num_classes, class_names = get_classes_info(yaml_file_path)
        return jsonify({'num_classes': num_classes}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/kill_training', methods=['POST'])
def kill_training():
    global training_process, training_status
    if training_process:
        try:
            training_status = False
            training_process.terminate()
            training_process.wait()
            training_process = None
            return jsonify({"message": "Training process killed"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No training process running"}), 400


@app.route('/training_true_false', methods=['GET'])
def training_true_false():
    try:
        #socketio.emit('training_status', {'training_status': training_status})
        return jsonify({'training_status': training_status}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/barcodes', methods=['GET'])
def get_barcodes():
    barcodes = load_barcodes()
    return jsonify(barcodes), 200

@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    receiver_url = 'http://ip:port/receive_file' #for this, type in the receiver code and the weights will go to that folder
    global weights_path
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    weights_path = data.get("weights_path")
    if not weights_path:
        return jsonify({"error": "No weights path provided"}), 400

    try:
        with open(weights_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(receiver_url, files=files)

        if response.status_code == 200:
            return jsonify({"message": "File sent successfully"}), 200
        else:
            return jsonify({"error": "Failed to send file"}), response.status_code
    except requests.RequestException as e:
        return jsonify({"error": f"Error sending file: {e}"}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('training_update')
def handle_training_update(data):
    emit('training_status', data, broadcast=True)

@socketio.on('barcodes_info_epochs')
def handle_training_update(data):
    emit('barcodes_info', data, broadcast=True)

@socketio.on('request_data')
def handle_request_data(data):
    print(f'Request data received: {data}')
    response_data = {
        'remaining_seconds': training.remaining_seconds,
        'completion_percentage': training.completion_percentage
    }

    response_data_2 = {
        'barcodes': load_barcodes()
    }

    emit('training_status', response_data)
    emit('barcodes_info', response_data_2)




if __name__ == '__main__':
    socketio.run(app, host='ip', port=port, debug=True, allow_unsafe_werkzeug=True)
