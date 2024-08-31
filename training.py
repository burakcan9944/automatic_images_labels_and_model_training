import os
import subprocess
import re
import time
import socketio
from yolov5 import train
import pathlib
import requests
import torch
import cv2

sio = socketio.Client()
remaining_seconds = 0
completion_percentage = 0


def read_barcodes_from_txt(txt_file_path):
    barcodes = []
    try:
        with open(txt_file_path, 'r') as file:
            for line in file:
                barcodes.append(line.strip())
    except FileNotFoundError:
        print(f"Error: The file {txt_file_path} does not exist.")
    return barcodes


def send_barcodes_via_socketio(barcode_info_list):
    for barcodes_info in barcode_info_list:
        sio.emit('barcodes_info_epochs', barcodes_info)


def train_model(epochs, batch_size, data, weights, save_dir, device):
    start_time = time.time()
    sio.connect('http://ip:port')

    try:
        weights_path = weights
        print(f"Using weights: {weights_path}")

        #yolov5 should be in the same folder as train.py
        train_cmd = [
            'python', 'train.py',
            '--data', data,
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--weights', weights_path,
            '--device', device,
            '--project', save_dir,
            '--name', '',
            '--exist-ok'
        ]

        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )

        time_pattern = re.compile(r'<(\d+:\d+),')
        epoch_pattern = re.compile(r'(\d+)/(\d+).*\s+(\d+)%\|')

        total_elapsed_seconds = 0
        epoch_count = 0
        previous_epoch = -1

        for line in process.stdout:
            print(line, end='')

            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                completion_percentage = ((current_epoch + 1) / epochs) * 100

                if current_epoch != previous_epoch:
                    match = time_pattern.search(line)
                    if match:
                        elapsed_time = match.group(1)
                        elapsed_seconds = sum(x * int(t) for x, t in zip([60, 1], elapsed_time.split(":")))

                        total_elapsed_seconds += elapsed_seconds
                        epoch_count += 1

                        previous_epoch = current_epoch

                        average_epoch_duration = total_elapsed_seconds / epoch_count

                        estimated_total_time = average_epoch_duration * epochs
                        estimated_remaining_time = estimated_total_time - total_elapsed_seconds
                        remaining_seconds = estimated_remaining_time

                        print(f'Epoch {current_epoch}/{total_epochs}, Completion: {completion_percentage:.2f}%')
                        print(f'Estimated remaining time: {remaining_seconds:.0f} seconds')

                        sio.emit('training_update', {
                            'completion_percentage': completion_percentage,
                            'remaining_seconds': remaining_seconds
                        })
                        barcode_info_txt_path = 'barcode_info.txt'
                        barcode_info_list = read_barcodes_from_txt(barcode_info_txt_path)
                        send_barcodes_via_socketio(barcode_info_list)


    except Exception as e:
        print(f"An error occurred during training: {e}")
        sio.emit('training_error', {'error': str(e)})
        return False

    return True


def evaluate_model(weights_folder, video_path):
    pathlib.PosixPath = pathlib.WindowsPath

    weights_files = os.listdir(weights_folder)
    if not weights_files:
        print(f"No weights files found in {weights_folder}")
        return

    weights_files.sort()
    weights_path = os.path.join(weights_folder, weights_files[0])

    print(f"Loading model with weights: {weights_path}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    high_confidence_detections = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")
        results = model(frame)
        df = results.pandas().xyxy[0]

        print(df)

        if df.empty:
            print(f"Frame {frame_count}: No detections")
        else:
            for _, row in df.iterrows():
                confidence = row['confidence']
                high_confidence_detections.append(confidence)

        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if high_confidence_detections:
        average_confidence = sum(high_confidence_detections) / len(high_confidence_detections)
    else:
        average_confidence = 0

    print(f"Average Confidence: {average_confidence}")

    if average_confidence > 0.8:
        high_confidence_detections.clear()
        data_send = {"weights_path": weights_path}
        try:
            response = requests.post('http://ip:port/process_video', json=data_send)
            if response.status_code == 200:
                print("Data sent successfully!")
            else:
                print("Failed to send data.")
        except requests.exceptions.RequestException as e:
            print(f"Error sending data: {e}")


if __name__ == "__main__":

    epochs = 10
    batch_size = 8
    data = 'barcodes.yaml'
    weights = 'yolov5s.pt'
    save_dir = 'runs\\train\\exp19'
    device = '0'

    training_successful = train_model(epochs, batch_size, data, weights, save_dir, device)

    if training_successful:
        weights_folder = os.path.join(save_dir, 'weights')
        video_path = 'video_1.mp4'
        evaluate_model(weights_folder, video_path)
