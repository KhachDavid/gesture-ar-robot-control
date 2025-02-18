import asyncio
import cv2
import glob
import math
import os
import subprocess
from importlib.resources import files

from matplotlib import pyplot as plt

from PIL import Image, ImageOps

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ROS2Controller import ROS2Controller

from frame_ble import FrameBle
from frame_msg import RxPhoto, TxCaptureSettings, TxSprite, TxImageSpriteBlock

# Initialize mediapipe gesture recognition
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

DOG_FRAMES_DIR = "dog_frames"
REMOTE_HOST = "192.168.12.33"  # IP of computer running the camera
UDP_PORT = "9000"


async def main():
    await check_camera_feed()

async def check_camera_feed():
    frame = FrameBle()
    
    try:
        await frame.connect()
        await frame.send_break_signal()

        # Let the user know we're starting
        await frame.send_lua("frame.display.text('Loading...',1,1);frame.display.show();print(1)", await_print=True)

        # debug only: check our current battery level
        print(f"Battery Level: {await frame.send_lua('print(frame.battery_level())', await_print=True)}")

        # send the std lua files to Frame that handle data accumulation and camera
        for stdlua in ['data', 'camera', 'image_sprite_block']:
            await frame.upload_file_from_string(files("frame_msg").joinpath(f"lua/{stdlua}.min.lua").read_text(), f"{stdlua}.min.lua")

    except Exception as e:
        print(f"Error: {e}")
        
    # Send the main lua application from this project to Frame that will run the app
    # to display the text when the messages arrive
    # We rename the file slightly when we copy it, although it isn't necessary
    await frame.upload_file("lua/camera_sprite_frame_app.lua", "frame_app.lua")

    # attach the print response handler so we can see stdout from Frame Lua print() statements
    # If we assigned this handler before the frameside app was running,
    # any await_print=True commands will echo the acknowledgement byte (e.g. "1"), but if we assign
    # the handler now we'll see any lua exceptions (or stdout print statements)
    frame._user_print_response_handler = print

    # "require" the main lua file to run it
    # Note: we can't await_print here because the require() doesn't return - it has a main loop
    await frame.send_lua("require('frame_app')", await_print=False)

    # give Frame a moment to start the frameside app,
    # based on how much work the app does before it's ready to process incoming data
    await asyncio.sleep(0.5)

    # Now that the Frameside app has started there is no need to send snippets of Lua
    # code directly (in fact, we would need to send a break_signal if we wanted to because
    # the main app loop on Frame is running).
    # From this point we do message-passing with first-class types and send_message() (or send_data())
    rx_photo = RxPhoto()
    await rx_photo.start()

    # hook up the RxPhoto receiver
    frame._user_data_response_handler = rx_photo.handle_data

    # give the frame some time for the autoexposure loop to run (50 times; every 0.1s)
    await asyncio.sleep(5.0)

    # start the photo capture loop and the ros2 controlling interface
    running = True  # Variable to control photo capture loop
    ros_controller = ROS2Controller(workspace_path="~/unitree_ws")

    images = []
    results = []

    i = 0  # Counter for photos
    while running:  # Keep capturing photos
        
         # Create async tasks for parallel execution
        task_gesture = asyncio.create_task(capture_and_recognize_gesture(frame, rx_photo, ros_controller, images, results, i))
        task_dog_frame = asyncio.create_task(fetch_and_display_dog_frame(frame))

        # Run both tasks in parallel and wait for them both to finish
        await asyncio.gather(task_gesture, task_dog_frame)

        i += 1
        await asyncio.sleep(0.1)  # Small delay

    # stop the photo handler and clean up resources
    rx_photo.stop()
    frame._user_data_response_handler = None

    display_batch_of_images_with_gestures_and_hand_landmarks(images, results)

    await frame.send_break_signal()
    if frame.is_connected():
        await frame.disconnect()


async def capture_and_recognize_gesture(frame, rx_photo, ros_controller, images, results, i):
    ########### TASK ONE START ###########
    # Request the photo capture
    capture_settings = TxCaptureSettings(0x0d, resolution=720)
    await frame.send_message(0x0d, capture_settings.pack())

    # get the jpeg bytes as soon as they're ready
    jpeg_bytes = await asyncio.wait_for(rx_photo.queue.get(), timeout=10.0)
    print(f"Received {len(jpeg_bytes)} bytes of jpeg data")

    # save the jpeg bytes to a file
    photo_filename = f"test_photo_{i}.jpg"
    with open(photo_filename, "wb") as f:
        f.write(jpeg_bytes)

    # Load the photo
    image = mp.Image.create_from_file(photo_filename)

    # Rotate the image
    recognition_result = recognizer.recognize(image)

    if not recognition_result.gestures:
        print(f"No gestures detected in photo {i + 1}")
        ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "none"}')
        await display_latest_dog_frame(frame)
        return

    images.append(image)
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    results.append((top_gesture, hand_landmarks))
    ########### TASK ONE END ###########

    # Publish to ROS Topic
    await handle_gesture_command(top_gesture.category_name.lower(), frame, ros_controller)


async def fetch_and_display_dog_frame(frame):
    """Fetches the latest image from the Unitree robot and displays it on Frame."""
    transfer_latest_image_from_robot()  # Fetch the image

    # Display it on Frame
    await display_latest_dog_frame(frame)

# Move this function to handle different gestures
async def handle_gesture_command(gesture, frame, ros_controller):
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


async def send_in_chunks(frame: FrameBle, msg_code, payload):
    """Send a large payload in BLE-compatible chunks."""
    max_chunk_size = frame.max_data_payload() - 5  # Maximum BLE payload size is 240
    print(f"Max BLE payload size: {max_chunk_size}")

    total_size = len(payload)  # Total size of the payload
    sent_bytes = 0  # Tracks how many bytes have been sent so far

    while sent_bytes < total_size:
        remaining_bytes = total_size - sent_bytes  # Remaining data to send
        chunk_size = min(max_chunk_size, remaining_bytes)  # Ensure ≤ max_chunk_size

        # Extract the next chunk
        chunk = payload[sent_bytes : sent_bytes + chunk_size]

        print(f"Sending chunk: {len(chunk)} bytes (offset: {sent_bytes}/{total_size})")

        # Add the msg_code (as the first byte of the packet) to the chunk
        if sent_bytes == 0:
            # first packet also has total payload length
            chunk_with_msg_code = bytearray([msg_code, total_size >> 8, total_size & 0xFF]) + chunk
        else:
            chunk_with_msg_code = bytearray([msg_code]) + chunk

        # Send the chunk
        await frame.send_data(chunk_with_msg_code)
        sent_bytes += chunk_size

        # Optional: Small delay to avoid overwhelming BLE
        await asyncio.sleep(0.01)

    print("All chunks sent successfully!")


# Fetch last captured image from dog_frames
async def display_latest_dog_frame(f):
    """Displays the last saved image from the local dog_frames directory."""
    try:
        # Get a list of all .jpg files in the directory
        jpg_files = glob.glob(os.path.join(DOG_FRAMES_DIR, "*.jpg"))

        if jpg_files:
            latest_file = max(jpg_files, key=os.path.getmtime)
            img = Image.open(latest_file)
            img = ImageOps.exif_transpose(img)  # Automatically correct rotation based on EXIF metadata

            # Get the first half of the image's width not the height because it is a stereo image
            img = img.crop((0, 0, img.width // 2, img.height))

            # Pack the image into a TxSprite object
            sprite = TxSprite.from_image_bytes(0x20, img, max_pixels=64000)
            isb = TxImageSpriteBlock(0x20, sprite, 20)
            await f.send_message(isb.msg_code, isb.pack())
            for spr in isb.sprite_lines:
                await f.send_message(isb.msg_code, spr.pack())

            print(f"Displaying latest image: {latest_file}")
        else:
            print("No images found in dog_frames.")
    
    except Exception as e:
        print(f"Error while displaying the image: {e}")


def transfer_latest_image_from_robot():
    """Transfers the latest image from the robot to `dog_frames/` using UDP."""
    # Use UDP to copy the latest file
    try:
        with open(DOG_FRAMES_DIR, "wb") as outfile:
            # Use netcat (nc) in UDP mode to connect to the remote server
            # and write the incoming bytes to outfile
            subprocess.run(["nc", "-u", REMOTE_HOST, UDP_PORT],
                           check=True,
                           stdout=outfile)
        print(f"Successfully transferred latest image via UDP to {DOG_FRAMES_DIR}.")
    except subprocess.CalledProcessError as e:
        print(f"UDP transfer failed: {e}")


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
    # rotate images back to original orientation
    images = [cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) for image in images]
    
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def rotate_image(image, angle):
    """Rotates a given Mediapie image by a specified angle."""
    # If the input image is in the form of a NumPy array, process it directly
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    # Get height and width of mediapipe image
    h, w, _ = image.numpy_view().shape

    # Compute the center of the image
    center = (w // 2, h // 2)

    # Create a rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Perform the rotation for mp.Image
    rotated_image_np = cv2.warpAffine(image.numpy_view(), M, (w, h))

    rotated_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rotated_image_np)

    # Return the rotated image as a NumPy array
    return rotated_image


if __name__ == "__main__":
    asyncio.run(main())
