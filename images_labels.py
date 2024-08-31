import os
import shutil
import random
import time
import torch
from yolov5 import train
import cv2
import pathlib
import requests
import yaml
import socketio
import argparse

sio = socketio.Client()
sio.connect('http://ip:port')

I_FILE = 'i_value.txt'

def read_i_value():
    if os.path.exists(I_FILE):
        with open(I_FILE, 'r') as f:
            return int(f.read().strip())
    else:
        return 0

def save_i_value(i):
    with open(I_FILE, 'w') as f:
        f.write(str(i))

def read_barcode_len():
    try:
        with open('barcode_len.txt', 'r') as f:
            barcode_len = int(f.read())
            return barcode_len
    except FileNotFoundError:
        return None

def read_urun_var_yok(var_yok_weights_path, rtsp):
    weights_path = var_yok_weights_path
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        rtsp_urun = rtsp
        cap = cv2.VideoCapture(rtsp_urun)
        detection_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            df = results.pandas().xyxy[0]

            for _, row in df.iterrows():
                name = row['name']
                class_id = int(row['class'])
                print(class_id)
                if class_id == 0:
                    detection_counter += 1
                    print(f"Detected {name} at frame {detection_counter}")
                    if detection_counter >= 10:
                        return True

            if df.empty:
                return False

            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def read_barcodes_from_file_append():
    with open('barcodes_append.txt', 'r') as f:
        lines = f.readlines()
        barcodes = [line.replace("Read barcode: ", "").strip() for line in lines]
        return barcodes

def process_rtsp_stream(rtsp_url, barcode, num_frames,i, start_row=100, start_col=150, end_row=300, end_col=400, base_dir='barcodes'):
    barcode_dir = os.path.join(base_dir, barcode)
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("RTSP stream could not be opened.")
        return

    os.makedirs(barcode_dir, exist_ok=True)

    train_images_dir = os.path.join(barcode_dir, 'train', 'images')
    train_labels_dir = os.path.join(barcode_dir, 'train', 'labels')
    valid_images_dir = os.path.join(barcode_dir, 'valid', 'images')
    valid_labels_dir = os.path.join(barcode_dir, 'valid', 'labels')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    frame_count = 0
    frames = []

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(barcode_dir, f'frame_{barcode}_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)

        h, w, _ = frame.shape
        yolo_label_filename = os.path.join(barcode_dir, f'frame_{barcode}_{frame_count:04d}.txt')
        center_x = (start_col + (end_col - start_col) / 2) / w
        center_y = (start_row + (end_row - start_row) / 2) / h
        width = (end_col - start_col) / w
        height = (end_row - start_row) / h
        yolo_label = f"{i} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"

        with open(yolo_label_filename, 'w') as f:
            f.write(yolo_label)

        frames.append((frame_filename, yolo_label_filename))
        frame_count += 1

    cap.release()

    random.shuffle(frames)

    split_idx = int(num_frames * 0.8)
    train_frames = frames[:split_idx]
    valid_frames = frames[split_idx:]

    for frame, label in train_frames:
        shutil.move(frame, os.path.join(train_images_dir, os.path.basename(frame)))
        shutil.move(label, os.path.join(train_labels_dir, os.path.basename(label)))

    for frame, label in valid_frames:
        shutil.move(frame, os.path.join(valid_images_dir, os.path.basename(frame)))
        shutil.move(label, os.path.join(valid_labels_dir, os.path.basename(label)))

    print(f"{frame_count}")
    print(f"{barcode} created Folder successfully.")

def read_barcodes_from_file_write():
    with open('barcodes_write.txt', 'r') as f:
        lines = f.readlines()
        barcodes = [line.replace("Read barcode: ", "").strip() for line in lines]
        return barcodes

def create_yaml_file(base_dir, barcodes, yaml_filename='barcodes.yaml'):
    base_path = 'base_path' #should be the path to the folder with all your codes
    train_images_dir = os.path.join(base_path, base_dir, "consolidated", 'train', 'images')
    valid_images_dir = os.path.join(base_path, base_dir, "consolidated", 'valid', 'images')

    data = {
        'train': train_images_dir,
        'val': valid_images_dir,
        'names': {i: barcodes[i] for i in range(len(barcodes))}
    }

    with open(yaml_filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)

    print(f"YAML file '{yaml_filename}' created:")
    print(data)

def consolidate_data(base_dir, barcodes):
    consolidated_train_images_dir = os.path.join(base_dir, 'consolidated', 'train', 'images')
    consolidated_train_labels_dir = os.path.join(base_dir, 'consolidated', 'train', 'labels')
    consolidated_valid_images_dir = os.path.join(base_dir, 'consolidated', 'valid', 'images')
    consolidated_valid_labels_dir = os.path.join(base_dir, 'consolidated', 'valid', 'labels')

    os.makedirs(consolidated_train_images_dir, exist_ok=True)
    os.makedirs(consolidated_train_labels_dir, exist_ok=True)
    os.makedirs(consolidated_valid_images_dir, exist_ok=True)
    os.makedirs(consolidated_valid_labels_dir, exist_ok=True)

    for barcode in barcodes:
        print(barcodes)
        barcode_dir = os.path.join(base_dir, barcode)
        train_images_dir = os.path.join(barcode_dir, 'train', 'images')
        train_labels_dir = os.path.join(barcode_dir, 'train', 'labels')
        valid_images_dir = os.path.join(barcode_dir, 'valid', 'images')
        valid_labels_dir = os.path.join(barcode_dir, 'valid', 'labels')

        for file in os.listdir(train_images_dir):
            shutil.copy(os.path.join(train_images_dir, file), os.path.join(consolidated_train_images_dir, file))
        for file in os.listdir(train_labels_dir):
            shutil.copy(os.path.join(train_labels_dir, file), os.path.join(consolidated_train_labels_dir, file))
        for file in os.listdir(valid_images_dir):
            shutil.copy(os.path.join(valid_images_dir, file), os.path.join(consolidated_valid_images_dir, file))
        for file in os.listdir(valid_labels_dir):
            shutil.copy(os.path.join(valid_labels_dir, file), os.path.join(consolidated_valid_labels_dir, file))





if __name__ == "__main__":

    i = read_i_value()

    barcodes = read_barcodes_from_file_write()
    for barcode in barcodes:
        barcode = barcode.strip()
        print(f"Read barcode: {barcode}")
        barcode_len = len(barcodes)
        rtsp_url='video_1.mp4'

        num_frames = 100

        var_yok_weights_path = 'best.pt' #this is the verification place, you can remove it
        var_yok_boolean = read_urun_var_yok(var_yok_weights_path, rtsp_url)
        print(f"var_yok_boolean: {var_yok_boolean}")

        if var_yok_boolean == True:
            print("product detected 10 times")
            continue

        if var_yok_boolean == False:
            process_rtsp_stream(rtsp_url, barcode, num_frames,i)
            print("i", i)
            i = i + 1
            save_i_value(i)

    base_dir = 'barcodes'
    consolidate_data(base_dir, barcodes)

    barcodes = read_barcodes_from_file_append()
    sio.emit('training_update', {'when_train_class': barcodes})

    create_yaml_file(base_dir, barcodes)
    print("Complate data consolidation")



