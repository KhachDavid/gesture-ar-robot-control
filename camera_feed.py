import asyncio
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
    f.ensure_connected()
    print(f"Connected: {f.bluetooth.is_connected()}")

    try:
        images = []
        results = []
        await f.display.write_text("Tap to start the camera feed test", align=Alignment.BOTTOM_CENTER)
        await f.motion.wait_for_tap()
        for i in range(50):  # Capture 50 photos for the test
            photo_filename = f"test_photo_{i}.jpg"
            print(f"Capturing photo {i + 1}...")
            # Capture a photo and save it to disk
            await f.camera.save_photo(photo_filename, autofocus_seconds=1, quality=Quality.HIGH, autofocus_type=AutofocusType.CENTER_WEIGHTED)
            print(f"Photo {i + 1} saved as {photo_filename}")
            # Load the photo from disk and display it
            image = mp.Image.create_from_file(photo_filename)
            # rotate the image to by 90 degrees counter-clockwise
            recognition_result = recognizer.recognize(image)
            # Display the recognition result
            # Error handling in case the recognition result is empty
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
            await f.display.write_text(f"Gesture: {top_gesture.category_name} ({top_gesture.score:.2f})", align=Alignment.MIDDLE_CENTER)
            # Add a small delay to simulate a consistent feed check
            await asyncio.sleep(1)
        display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
        print("Camera feed test completed successfully!")
    except Exception as e:
        print(f"An error occurred during the camera feed test: {e}")

    f.bluetooth.disconnect()


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


if __name__ == "__main__":
    asyncio.run(main())
