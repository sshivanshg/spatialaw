import socket

UDP_IP = "0.0.0.0"   # Listen on all network interfaces
UDP_PORT = 514       # Default syslog port

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"✅ Listening for syslog messages on UDP port {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(4096)
    print(f"{addr[0]}: {data.decode(errors='ignore')}")