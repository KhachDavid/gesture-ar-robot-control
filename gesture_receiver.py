import time
import bluetooth
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ROS2Controller import ROS2Controller

# Initialize ROS2 controller
ros_controller = ROS2Controller()

# Initialize mediapipe gesture recognition
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

images = []
results = []

def receive_images(save_directory, port=1):
    try:
        # Ensure the save directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        server_sock.bind(("", port))  # Bind to all interfaces
        server_sock.listen(1)

        print("Waiting for connections...")

        image_counter = 1  # Keeps track of received images

        client_sock, address = server_sock.accept()
        while True:
            received_file = os.path.join(save_directory, f"image_{image_counter}.jpg")  
            with open(received_file, "wb") as image_file:
                while True:
                    data = client_sock.recv(1024)  # Receive data in chunks
                    if not data or b"END_OF_IMAGE" in data:  # Detect transfer end marker
                        break
                    image_file.write(data)

            print(f"Image received and saved as {received_file}")

            # Attempt to load and process the image
            try:
                image = mp.Image.create_from_file(received_file)
                recognition_result = recognizer.recognize(image)
                
                if not recognition_result.gestures:
                    print(f"No gestures detected in photo {image_counter}")
                    ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "none"}')
                    image_counter += 1  # Increment filename for the next image
                    continue

                top_gesture = recognition_result.gestures[0][0]
                handle_gesture_command(top_gesture.category_name.lower(), ros_controller)
            
            except Exception as e:
                print(f"Skipping corrupted image {received_file}: {e}")

            image_counter += 1  # Increment count for the next image

    except Exception as e:
        print(f"Error in receiving images: {e}")
    finally:
        server_sock.close()

# Handle gestures
def handle_gesture_command(gesture, ros_controller):
    """Handles the corresponding action based on detected gesture."""
    GESTURE_COMMANDS = {
        "thumb_up": "up",
        "thumb_down": "down",
        "pointing_up": "forward",
        "victory": "back",
        "iloveyou": "right",
        "open_palm": "left",
        "closed_fist": "hand"
    }
    
    command = GESTURE_COMMANDS.get(gesture, "none")
    print(f"Executing command: {command}")
    ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", f'{{data: "{command}"}}')


if __name__ == "__main__":
    save_directory = "received_images"
    receive_images(save_directory)
