import socket

# Remote IP/Port to send the log data to (TCP)
RHOST = '192.168.0.163'
RPORT = 23

def sendLog(fromip, message):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((RHOST, RPORT))
    s.send(('IP:' + fromip + ' Port:' + str(LPORT) + ' | ' + message.replace(b'\r\n', b' ')).encode())
    s.close()

if __name__ == '_main_':
    # Example usage
    fromip = '192.168.0.5'  # Replace with the actual IP address
    message = 'Some log data'  # Replace with the actual log data
    sendLog(fromip, message)