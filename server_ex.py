import io
import socket
import struct
import threading
from PIL import Image

def main() :
    IP = "141.223.122.36"
    CAPTURE_PORT = 8100
    LABEL_PORT = 8000

    #def send_label(label, socket) :
    #    socket.send(label.encode())


    # Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
    # all interfaces)
    s_capture_socket = socket.socket()
    print("capture socket create")

    s_capture_socket.bind(("", CAPTURE_PORT))
    print("capture socket bind")

    s_capture_socket.listen(5)
    print("capture socket listen")


    s_label_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("label socket create")

    s_label_socket.bind(("", LABEL_PORT))
    print("label socket bind")

    s_label_socket.listen(5)
    print("label socket listen")

    #th = threading.Thread(target = send_label)

    num = 0

    while True :
        # Accept a single connection and make a file-like object out of it
        capture_connection = s_capture_socket.accept()[0].makefile('rb')
        label_connection = s_label_socket.accept()[0]

        try:
             while True:
                # Read the length of the image as a 32-bit unsigned int. If the
                # length is zero, quit the loop
                image_recv = capture_connection.read(struct.calcsize('<L'))
                if image_recv == 0 :
                    return

                image_len = struct.unpack('<L', image_recv)[0]

                if not image_len:
                    return

                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = io.BytesIO()
                image_stream.write(capture_connection.read(image_len))

                # Rewind the stream, open it as an image with PIL and do some
                # processing on it
                image_stream.seek(0)
                image = Image.open(image_stream)
                dataStr = "image%2s.jpg receive" % num
                image.save("train_image/image%02s.jpg" % num)
                num += 1

                label_connection.send(dataStr.encode())

                print('Image is verified')

        except KeyboardInterrupt :
            capture_connection.close()
            s_capture_socket.close()
            s_label_socket.close()

            return 0


if __name__ == "__main__" :
    main()
