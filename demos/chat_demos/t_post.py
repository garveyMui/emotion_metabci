import numpy as np
import time
import socket
def post():
    labels = np.random.random((1, 3))
    emotion_dict = {0: "sad", 1: "neutral", 2: "happy"}
    emotion_key = np.argmax(labels)
    emotion = emotion_dict[emotion_key]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('127.0.0.1', 4023)
    messages = emotion
    try:
        while True:
            if time.time() % 1 == 0:
                print(f'Sending "{messages}" to {server_address}')
            sock.sendto(messages.encode(), server_address)
            response, server_address = sock.recvfrom(1024)
            if response.decode() == "got it":
                print("Send successfully")
                break
    finally:
        print('Closing socket')
        sock.close()
post()