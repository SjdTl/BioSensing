# echo-client.py
#%%
import json
import socket

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 13854  # The port used by the server
config = json.dumps({"appName":"Brainwave Test","appKey":"0139ccebc1902e0905b11bebc63c82eecada5784"})
config = config.encode('utf-8')
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(config)
    data = s.recv(2**32)

print(f"Received {data!r}")
# print(f"Received {json.loads(data.decode('utf-8'))}")
# %%
