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


# Initialize mediapipe gesture recognition
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

async def main():
    await check_camera_feed()

async def check_camera_feed():
    f = Frame()
    try:
        await f.ensure_connected()
    except Exception as e:
        print(f"An error occurred while connecting to the Frame: {e}")
        await f.ensure_connected()

    print(f"Connected: {f.bluetooth.is_connected()}")

    running = True  # Variable to control photo capture loop

    def stop_on_tap():
        """Callback to stop capturing photos when a tap is detected."""
        nonlocal running
        running = False

    try:
        images = []
        results = []
        #await f.display.show_text("Tap to start capturing photos", align=Alignment.BOTTOM_CENTER)
        #await f.motion.wait_for_tap()
        await f.display.write_text("Capturing… Tap to stop", align=Alignment.BOTTOM_CENTER)
        battery_level = await f.get_battery_level()
        # Register the tap handler to stop photo capture using `run_on_tap`
        # This sets up the callback to stop the loop when a tap is detected.
        await f.motion.run_on_tap(callback=stop_on_tap)
        print(f"Battery level: {battery_level}%")
        await f.display.write_text(f"Battery level: {battery_level}%", align=Alignment.TOP_RIGHT)
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
                continue

            if not recognition_result.hand_landmarks:
                print(f"No hand landmarks detected in photo {i + 1}")
                continue

            images.append(image)
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks
            results.append((top_gesture, hand_landmarks))
            print(f"Recognition result for photo {i + 1}: {top_gesture.category_name} ({top_gesture.score:.2f})")
            await f.display.write_text(f"Gesture: {top_gesture.category_name} ({top_gesture.score:.2f})")

            # Increment the photo counter
            i += 1

            # Add a small delay between captures
            await asyncio.sleep(2)

        # Once the user stops capturing photos, display the batch of images with results
        display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
        print("Photo capture session completed successfully!")

    except Exception as e:
        print(f"An error occurred during the photo capture: {e}")

    if f.bluetooth.is_connected():
        await f.bluetooth.disconnect()

def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
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
