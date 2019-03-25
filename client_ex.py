import io
import socket
import struct
import time
import picamera
import threading

IP = "141.223.140.36"
CAPTURE_PORT = 8100
LABEL_PORT = 8000

def recv_label() :
    global c_label_socket

    while True :
        recvStr = c_label_socket.recv(1024)
        if len(recvStr) < 0 :
            print("shit")
        
        elif len(recvStr) == 0 :
            break

        print(recvStr.decode())


# Socket to send capture image
c_capture_socket = socket.socket()
print("capture socket create")
c_capture_socket.connect((IP, CAPTURE_PORT))
print("capture socket connected")

# Socket to recv label
c_label_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("label socket create")
c_label_socket.connect((IP, LABEL_PORT))
print("label socket connected")

# recv thread
th = threading.Thread(target = recv_label)
th.start()

# Make a file-like object out of the connection
connection = c_capture_socket.makefile('wb')

try:
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 20

        # Start a preview and let the camera warm up for 2 seconds
        camera.start_preview()
        time.sleep(2)
                     
        # Note the start time and construct a stream to hold image data
        # temporarily (we could write it directly to connection but in this
        # case we want to find out the size of each capture first to keep
        # our protocol simple)
        start = time.time()
        stream = io.BytesIO()

        for foo in camera.capture_continuous(stream, 'jpeg', use_video_port = True):
            # Write the length of the capture to the stream and flush to                 
            # ensure it actually gets sent

            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            
            # Rewind the stream and send the image data over the wire
            stream.seek(0)                                             
            connection.write(stream.read())
            
            # 일단 5초동안만 찍기
            if time.time() - start > 1 :
                break
                
            # Reset the stream for the next capture
            stream.seek(0)
            stream.truncate()
                          
            # Write a length of zero to the stream to signal we're done
            # connection.write(struct.pack('<L', 0))


finally:
    connection.close()
    c_capture_socket.close()
    c_label_socket.close()
