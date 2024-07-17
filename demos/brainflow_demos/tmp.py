import socket
def post_test():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    local_ip = '0.0.0.0'
    local_port = 4022
    # sock.bind((local_ip, local_port))
    server_address = ('10.1.124.74', 4023)
    messages = "我今天被领导骂了一顿，好想哭。"
    try:
        print(f'Sending "{messages}" to {server_address}')
        sent = sock.sendto(messages.encode(), server_address)

    finally:
        print('Closing socket')
        sock.close()
