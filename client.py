import socketio
import time


sio = socketio.Client()

sio.connect('http://ip:port')

@sio.event
def connect():
    print('Connected to server')

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.on('training_status')
def on_training_status(data):
    print('Training status update:', data)

@sio.on('barcodes_info')
def barcodes_info(data):
    print('barcodes_info:', data)

@sio.on('request_data')
def handle_request_data(sid, data):
    print(f'Data request received from {sid}: {data}')


def main():
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sio.disconnect()

if __name__ == '__main__':
    main()
