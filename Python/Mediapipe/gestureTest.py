import mediapipe as mp
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# mp_drawing = mp.solutions.drawing_utils
# landmarker = mp.tasks.vision.HandLandmarker

cap = cv2.VideoCapture(0)


# import mediapipe as mp

model_path = 'gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    r = result.gestures
    if r:
        print('gesture recognition result: {}'.format(r))

        if len(r) == 1:
            print("One Gesture: {}".format(r[0][0].category_name))
            print(type(result.hand_landmarks[0][0]))
            # for l in result.hand_landmarks:
            #     print(type(l))
                # print("Landmarks: {}".format(l))
        elif len(r) == 2:
            print("Two Gestures: {} + {}".format(r[0][0].category_name, r[1][0].category_name))

        # for res in r[0]:
        #     print(res)
        # if len(r) > 1:
        #     print(r[1][0].category_name)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2)

recognizer = GestureRecognizer.create_from_options(options)
  # The detector is initialized. Use it here.
  # ...
while cap.isOpened():
    ret, frame = cap.read()
    m = cap.get(cv2.CAP_PROP_POS_MSEC)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if cv2.waitKey(5) == ord('q'):
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    r = recognizer.recognize_async(mp_image, int(m))


    cv2.imshow('Holistic Model Detection', frame)



    # # STEP 3: Load the input image.
    # image = mp.Image.create_from_file("image.jpg")

    # # STEP 4: Detect hand landmarks from the input image.
    # detection_result = detector.detect(image)

    # # STEP 5: Process the classification result. In this case, visualize it.
    # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    



cap.release()
cv2.destroyAllWindows()