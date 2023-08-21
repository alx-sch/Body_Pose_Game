import cv2
import mediapipe as mp
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


def predict_pose(landmark_coordinates, label_mapping, model):
    landmark_coordinates_flattened = landmark_coordinates.reshape(1, -1)
    prediction = model.predict(landmark_coordinates_flattened)
    predicted_pose_index = np.argmax(prediction)
    pred_prob = prediction[0][predicted_pose_index]

    predicted_label = next(
        (label for label, idx in label_mapping.items() if idx == predicted_pose_index),
        "Pose",
    )

    if pred_prob < 0.7:
        predicted_label = "Pose"

    return predicted_label


def predict_pose_2(landmark_coordinates, label_mapping, model):
    landmark_coordinates_flattened = landmark_coordinates.reshape(1, -1)
    prediction = model.predict(landmark_coordinates_flattened)

    predicted_pose_index = np.argmax(prediction)
    predicted_pose_index_2 = np.argsort(prediction[0])[-2]

    pred_prob = prediction[0][predicted_pose_index]

    predicted_label = next(
        (label for label, idx in label_mapping.items() if idx == predicted_pose_index),
        "Pose",
    )
    predicted_label_2 = next(
        (
            label
            for label, idx in label_mapping.items()
            if idx == predicted_pose_index_2
        ),
        "Pose",
    )

    if predicted_label == "Stand" and pred_prob > 0.5:
        predicted_label = "Stand"
    elif (
        predicted_label == "Pose"
        and predicted_label_2 == "Stand"
        and prediction[0][np.argsort(prediction[0])[-2]] > 0.2
    ):
        predicted_label = "Stand"
    elif pred_prob < 0.7 and predicted_label != "Stand":
        predicted_label = "Pose"

    return predicted_label
