from socket import *
def collecting_all_info(user_name, user_info):
    udp_port_Online_emotion = 4023
    tcp_port_chatchat = 4024

    Online_emotion_server = socket(AF_INET, SOCK_DGRAM)
    chatchat_server = socket(AF_INET, SOCK_DGRAM)

    Online_emotion_server.bind(('127.0.0.1', udp_port_Online_emotion))
    chatchat_server.bind(('127.0.0.1', tcp_port_chatchat))
    # chatchat_server.listen(5)

    while True:
        emotion, clientAddress = Online_emotion_server.recvfrom(1024)
        Online_emotion_server.sendto("got it".encode(), clientAddress)
        if emotion:
            Online_emotion_server.close()
            break
    emotion = emotion.decode()
    print(emotion)
    while True:
        print("into listening")
        # connectionSocket, addr = chatchat_server.accept()
        message, addr = chatchat_server.recvfrom(1024)
        # message = chatchat_server.recv(1024)
        if message.decode() == "user_name":
            chatchat_server.send(user_name.encode(), addr)
        elif message.decode() == "user_info":
            chatchat_server.send(user_info.encode(), addr)

    print("finished")