import asyncio
import cv2
import math

from frame_sdk import Frame
from frame_sdk.camera import AutofocusType, Quality
from frame_sdk.display import Alignment

from matplotlib import pyplot as plt

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ROS2Controller import ROS2Controller
from frame_image.TxSprite import TxSprite
from PIL import Image
from bleak import BleakError

# Initialize mediapipe gesture recognition
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

async def main():
    #await check_camera_feed()
    f = Frame()
    try:
        await f.ensure_connected()
    except Exception as e:
        print(f"An error occurred while connecting to the Frame: {e}")
        await f.ensure_connected()

    print(f"Connected: {f.bluetooth.is_connected()}")
    temp_file = "test_photo_0.jpg"
    await process_and_send_image(f, temp_file)


async def send_in_chunks(f, msg_code, payload):
    """Send a large payload in BLE-compatible chunks."""
    max_chunk_size = f.bluetooth.max_data_payload() - 5  # Maximum BLE payload size is 240
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
        chunk_with_msg_code = bytearray([msg_code]) + chunk

        # Send the chunk
        await f.bluetooth.send_data(chunk_with_msg_code)
        sent_bytes += chunk_size

        # Optional: Small delay to avoid overwhelming BLE
        await asyncio.sleep(0.01)

    print("All chunks sent successfully!")


async def process_and_send_image(f: Frame, image_path: str):
    """Load a pre-existing image, process it, and send it to Frame in chunks."""
    
    print(f"Loading preloaded image: {image_path}")

    #  Load image and convert to indexed color mode (Palette Mode)
    img = Image.open(image_path).convert("RGB")  # Convert to RGB Mode
    img = img.convert("P", palette=Image.ADAPTIVE, colors=16)  # Convert to 16 colors
    img = img.resize((320, 200))  # Optionally resize for Frame compatibility
    
    # Save the processed image as PNG (for debugging or testing)
    processed_image_path = "processed_sprite.png"
    img.save(processed_image_path)
    print(f"Image processed and saved as: {processed_image_path}")

    # Pack the image into a TxSprite object
    sprite = TxSprite(msg_code=0x20, image_path=processed_image_path)
    packed_data = sprite.pack()

    # Check the size of the packed payload
    print(f"Packed sprite payload size: {len(packed_data)} bytes")

    # Send the packed data to Frame in chunks
    try:
        print("Sending image data in BLE-compatible chunks...")
        await send_in_chunks(f, sprite.msg_code, packed_data)
        print("Image successfully sent!")
    except Exception as e:
        print(f"Failed to send image: {e}")


async def handle_imu_motion_control(frame: Frame, ros_controller: ROS2Controller):
    await frame.motion.run_on_tap(callback=None)

    while True:
        try:
            while True:  # Continuously check IMU data
                try:
                    direction = await frame.motion.get_direction()
                    print(f"IMU Reading - Roll: {direction.roll}, Pitch: {direction.pitch}, Heading: {direction.heading}")
                    
                    if direction.roll > 10.0:  # Roll tilt to the right
                        await frame.display.show_text("Leaning Right!")
                        ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "right"}')
                    
                    elif direction.roll < -10.0:  # Roll tilt to the left
                        await frame.display.show_text("Leaning Left!")
                        ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "left"}')
    
                    elif direction.pitch > 15.0:  # Pitch tilt forward
                        await frame.display.show_text("Leaning Forward!")
                        ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "forward"}')
                    
                    elif direction.pitch < -15.0:  # Pitch tilt backward
                        await frame.display.show_text("Leaning Backward!")
                        ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "back"}')
    
                    else:
                        await frame.display.show_text("Standing Still!")
                        ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "none"}')
    
                except Exception as e:
                    print(f"An error occurred while reading IMU data: {e}")
                    break  # Exit the loop on repeated failure
        except Exception as e:
            print(f"IMU Error: {e}")
            break

async def check_camera_feed():
    f = Frame()
    try:
        await f.ensure_connected()
    except Exception as e:
        print(f"An error occurred while connecting to the Frame: {e}")
        await f.ensure_connected()

    print(f"Connected: {f.bluetooth.is_connected()}")

    running = True  # Variable to control photo capture loop
    ros_controller = ROS2Controller(workspace_path="~/unitree_ws")

    def stop_on_tap():
        """Callback to stop capturing photos when a tap is detected."""
        #nonlocal running
        #running = False

    try:
        #ros_controller.start_node("unitree_legged_real", "frames_open_palm")
        images = []
        results = []
        #await f.display.show_text("Tap to start capturing photos", align=Alignment.BOTTOM_CENTER)
        #await f.motion.wait_for_tap()
        battery_level = await f.get_battery_level()
        # Register the tap handler to stop photo capture using `run_on_tap`
        # This sets up the callback to stop the loop when a tap is detected.
        await f.motion.run_on_tap(callback=stop_on_tap)
        print(f"Battery level: {battery_level}%")
        await f.display.show_text(f"Battery level: {battery_level}%", align=Alignment.TOP_RIGHT)
        i = 0  # Counter for photos
        while running:  # Keep capturing photos until the tap handler stops the loop
            photo_filename = f"test_photo_{i}.jpg"
            print(f"Capturing photo {i + 1}...")

            # Capture a photo and save it to disk
            await f.camera.save_photo(photo_filename, autofocus_seconds=1, quality=Quality.HIGH, autofocus_type=AutofocusType.CENTER_WEIGHTED)
            print(f"Photo {i + 1} saved as {photo_filename}")

            # Load the photo
            image = mp.Image.create_from_file(photo_filename)

            # Rotate the image by 90 degrees counter-clockwise (if needed)
            recognition_result = recognizer.recognize(rotate_image(image, angle=90))

            # Check if any gestures or hand landmarks are detected
            if not recognition_result.gestures:
                print(f"No gestures detected in photo {i + 1}")
                await f.display.show_text("Nothing detected!")
                ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "none"}')

                direction = await f.motion.get_direction()
                print(f"IMU Reading - Roll: {direction.roll}, Pitch: {direction.pitch}, Heading: {direction.heading}")
                    
                #if direction.roll > 10.0:  # Roll tilt to the right
                #    await f.display.show_text("Leaning Right!")
                #    ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "right"}')
                #    
                #elif direction.roll < -10.0:  # Roll tilt to the left
                #    await f.display.show_text("Leaning Left!")
                #    ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "left"}')

                #elif direction.pitch > 15.0:  # Pitch tilt forward
                #    await f.display.show_text("Leaning Forward!")
                #    ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "forward"}')
                #
                #elif direction.pitch < -15.0:  # Pitch tilt backward
                #    await f.display.show_text("Leaning Backward!")
                #    ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "back"}')
                #else:
                #    await f.display.show_text("Standing Still!")
                #    ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "none"}')


                continue

            if not recognition_result.hand_landmarks:
                print("Nothing Detected!")
                await f.display.show_text("Nothing detected!")
                ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "none"}')

                print(f"No hand landmarks detected in photo {i + 1}")
                continue

            images.append(image)
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks
            results.append((top_gesture, hand_landmarks))
            print(f"Recognition result for photo {i + 1}: {top_gesture.category_name} ({top_gesture.score:.2f})")

            if (top_gesture.category_name.lower() == "thumb_up"):
                # invoke the robot to move up with ros2 run unitree_legged_real frames_open_palm
                print("Thumbs up detected!")
                await f.display.show_text("Going up!")
                ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "up"}')

            elif (top_gesture.category_name.lower() == "thumb_down"):
                # invoke the robot to move up with ros2 run unitree_legged_real frames_open_palm
                print("Thumbs down detected!")
                await f.display.show_text("Going to sleep!")
                ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "down"}')

            elif (top_gesture.category_name.lower() == "pointing_up"):
                # invoke the robot to move up with ros2 run unitree_legged_real frames_open_palm
                print("Pointing up detected!")
                await f.display.show_text("Going forward!")
                ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "forward"}')

            elif (top_gesture.category_name.lower() == "victory"):
                # invoke the robot to move up with ros2 run unitree_legged_real frames_open_palm
                print("Victory detected!")
                await f.display.show_text("Going back!")
                ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "back"}')

            elif (top_gesture.category_name.lower() == "iloveyou"):
                print("Open palm detected!")
                await f.display.show_text("Going right!")
                ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "right"}')

            elif (top_gesture.category_name.lower() == "open_palm"):
                print("Open palm detected!")
                await f.display.show_text("Going left!")
                ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "left"}')
            
            elif (top_gesture.category_name.lower() == "closed_fist"):
                print("Closed Fist detected!")
                await f.display.show_text("Fist!")
                ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "hand"}')

            
            else:
                print("Nothing Detected!")
                await f.display.show_text("Nothing detected!")
                ros_controller.publish_to_topic("/active_gesture", "std_msgs/msg/String", '{data: "none"}')
                
                # execute the robot movement using subprocess
                #ros2_worker.start_ros2_node("unitree_legged_real", "frames_open_palm")
            # Increment the photo counter
            i += 1

            # Add a small delay between captures
            await asyncio.sleep(0.1)

        # Once the user stops capturing photos, display the batch of images with results
        #ros_controller.kill_node()
        display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
        print("Photo capture session completed successfully!")

    except Exception as e:
        #ros_controller.kill_node()
        print(f"An error occurred during the photo capture: {e}")

    if f.bluetooth.is_connected():
        await f.bluetooth.disconnect()

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
