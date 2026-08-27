import mediapipe as mp
import cv2

from pythonosc.udp_client import SimpleUDPClient
from pythonosc import osc_message_builder
from pythonosc import osc_bundle_builder


PORT = 8080
IP = "127.0.0.1"
RESULTS = None


def setup_osc_client(ip, port):
    return SimpleUDPClient(ip, port)


def build_bundle(address, pos):
    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    msg = osc_message_builder.OscMessageBuilder(address=address)
    msg.add_arg(pos[0], arg_type='f')
    msg.add_arg(pos[1], arg_type='f')
    msg.add_arg(pos[2], arg_type='f')
    bundle.add_content(msg.build())

    return bundle.build()

def build_message(address, message):
    msg = osc_message_builder.OscMessageBuilder(address=address)
    msg.add_arg(message)
    return msg


def capture_and_process_webcam():
    global RESULTS
    global IP
    global PORT
    client = setup_osc_client(IP, PORT)
    cap = cv2.VideoCapture(0)

    model_path = 'C:/Users/parke/Documents/Coding/mpTest/hand_landmarker.task'

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a gesture recognizer instance with the live stream mode:
    def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        global RESULTS
        RESULTS = result

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=2)



    landmarker = HandLandmarker.create_from_options(options)

    # The detector is initialized. Use it here.
    # ...
    while cap.isOpened():

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        m = cap.get(cv2.CAP_PROP_POS_MSEC)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if cv2.waitKey(5) == ord('q'):
            break


        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        landmarker.detect_async(mp_image, int(m))


        # if RESULTS:
        if RESULTS is not None and len(RESULTS.hand_landmarks) > 0:
            bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)

            for hand_index in range(len(RESULTS.hand_landmarks)):
                hand = RESULTS.hand_landmarks[hand_index]
                hand_bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)

                for landmark_index in range(len(hand)):
                    landmark = hand[landmark_index]
                    print("{}, {}: {}".format(hand_index, landmark_index, landmark))

                    x = float(landmark.x)
                    y = float(landmark.y)
                    z = float(landmark.z)

                    h , w , _ = frame.shape
                    cv2.putText(frame, org=(int(x * w), int(y * h)), color=(0, 0, 255/21 * landmark_index), thickness=2, text=str(landmark_index),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5)
                    handname = "/hand/{}/{}".format(hand_index, landmark_index)

                    x_msg = build_message(handname + ".x", x)
                    y_msg = build_message(handname + ".y", y)
                    z_msg = build_message(handname + ".z", z)
                    bundle.add_content(x_msg.build())
                    bundle.add_content(y_msg.build())
                    bundle.add_content(z_msg.build())

            client.send(bundle.build())
        else:
            msg = build_message("/health", 1)
            bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            bundle.add_content(msg.build())
            client.send(bundle.build())

        cv2.imshow('Holistic Model Detection', frame)

        if cv2.waitKey(5) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_process_webcam()
    pass
