import cv2
import time
import mediapipe as mp
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

POSES = ["stand", "squat", "X", "crane", "empty"]

POSES = [POSES[-1]] + POSES[:-1]
COUNTDOWN = 4

ITERATIONS_PER_POSE = 10
ITERATIONS_PER_POSE = ITERATIONS_PER_POSE * len(POSES)

rect_top_left = (950, 20)
rect_bottom_right = (350, 700)
rect_color = (0, 255, 0)


def initialize_pose_model():
    return mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
    )


def process_frame(frame, pose):
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)

    # Draw the pose annotation on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=4, circle_radius=5
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 0, 0), thickness=15, circle_radius=5
        ),
    )
    return results, frame


def draw_bold_text(
    image,
    text,
    position,
    font_scale=2.5,
    color=(255, 255, 255),
    thickness=5,
    line_type=cv2.LINE_AA,
    offset=2,
):
    """
    Draw bolder text with slight offsets to create a thicker appearance.

    Args:
        image (numpy.ndarray): The image on which to draw the text.
        text (str): The text to be drawn.
        position (tuple): (x, y) position of the text.
        font_scale (float): Font scale.
        color (tuple): Text color (BGR format).
        thickness (int): Thickness of the text.
        line_type: Line type for drawing text.
        offset (int): Offset for creating bolder appearance.
    """
    for offset_x, offset_y in [
        (-offset, -offset),
        (-offset, offset),
        (offset, -offset),
        (offset, offset),
    ]:
        offset_position = (position[0] + offset_x, position[1] + offset_y)
        cv2.putText(
            image,
            text,
            offset_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            line_type,
        )


def extract_landmarks(results):
    landmark_list = results.pose_landmarks

    if landmark_list:
        filtered_landmarks = (
            landmark_list.landmark[11:17] + landmark_list.landmark[23:29]
        )
        coordinate_list = []
        for landmark in filtered_landmarks:
            x = landmark.x
            y_ = landmark.y
            coordinate_list.append([x, y_])
        coordinates = np.array(coordinate_list)
    else:
        coordinates = np.zeros((12, 2))

    return coordinates


"""
def load_landmark_list(filename):
    # Read the serialized data from the file
    with open(filename, "rb") as file:
        serialized_data = file.read()

    # Deserialize the data into a NormalizedLandmarkList object
    normalized_landmark_list = landmark_pb2.NormalizedLandmarkList()
    normalized_landmark_list.ParseFromString(serialized_data)

    return normalized_landmark_list
"""


def main():
    cap = cv2.VideoCapture(0)  # Open the camera

    last_capture_time = time.time()
    countdown = False
    count = 0  # Initialize count to 0
    iterations_left = ITERATIONS_PER_POSE
    current_pose_index = 0

    pose = initialize_pose_model()

    while True:
        success, frame = cap.read()

        if success:
            cv2.rectangle(
                frame, rect_top_left, rect_bottom_right, rect_color, thickness=4
            )

        if success:
            results, frame = process_frame(frame, pose)

            # mirror frame
            frame = cv2.flip(frame, 1)

            if countdown:
                if iterations_left > 0:
                    if (
                        count == 0
                    ):  # Initialize count to 5 when starting a new pose iteration
                        count = COUNTDOWN
                        current_pose_index = (current_pose_index + 1) % len(
                            POSES
                        )  # Move to the next pose

                    current_pose = POSES[current_pose_index]
                    round_no = int(np.ceil(iterations_left / len(POSES)))

                    # Display pose & countdown on frame:
                    text = f"{current_pose.upper()} (left: {round_no}): {count} s"
                    text_position = (30, 100)

                    draw_bold_text(frame, text, text_position, color=(0, 0, 255))

                    # Display the frame
                    cv2.imshow("Pose Detection", frame)

                    if time.time() - last_capture_time >= 1:
                        count -= 1
                        last_capture_time = time.time()

                    if count == 0:
                        picture_folder = f"test_marco/pictures/{current_pose}"
                        landmarks_folder = f"test_marco/landmarks/{current_pose}"
                        if not os.path.exists(picture_folder):
                            os.makedirs(picture_folder)
                        if not os.path.exists(landmarks_folder):
                            os.makedirs(landmarks_folder)
                        timestamp = int(time.time())

                        landmark_coordinates = extract_landmarks(results)

                        filename_pictures = (
                            f"{picture_folder}/{current_pose}_{timestamp}.jpg"
                        )
                        filename_landmarks = (
                            f"{landmarks_folder}/{current_pose}_{timestamp}.npy"
                        )

                        cv2.imwrite(filename_pictures, frame)
                        np.save(filename_landmarks, landmark_coordinates)

                        iterations_left -= 1

                else:
                    countdown = False
                    iterations_left = ITERATIONS_PER_POSE

            else:
                text = "Press SPACE to start countdown"
                text_position = (30, 100)

                draw_bold_text(
                    frame,
                    text,
                    text_position,
                    font_scale=2,
                    color=(255, 0, 0),
                    thickness=2,
                    offset=1,
                )

            # display frame / videostream in window
            cv2.imshow("Pose Detection", frame)

            key = cv2.waitKey(5)
            if key == ord("q"):  # Press 'q' to exit
                break

            if key == ord(" "):
                countdown = True
                last_capture_time = time.time()

        else:
            print("Error: Could not open camera.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
