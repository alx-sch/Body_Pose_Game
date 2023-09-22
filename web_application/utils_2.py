"""
UTILITY FUNCTIONS 
used to 1) collect training data (collect_data.py) 
and 2) to play the body pose game (play.py)
"""
##############
### IMPORT ###
##############

import cv2  # OpenCV for video capture and processing
import os  # Operating system-related functions, e.g. directory and path operations
import time  # Time-related function
import numpy as np  # NumPy for numerical computations

# Importing modules from mediapipe.solutions (for pose landmark detection)
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

##################
### PARAMETERS ###
##################

# Constants used in 'predict_pose' functions, which are used to accommodate the model's behavior as observed during testing (see training_model.ipynb).
CONFIDENCE_THRESHOLD = 0.7  # min. prediction confidence to be reached (but for 'Stand'), otherwise reclassified to 'Pose'.
CONFIDENCE_THRESHOLD_STAND = 0.5  # min. prediction confidence to be reached for 'Stand', otherwise reclassified to 'Pose'.
CONFIDENCE_THRESHOLD_POSE_TO_STAND = 0.2  # min. prediction confidence classification 'Stand' if 'Pose' is the most likel prediction with 'Stand' being the second most likely.

#################
### FUNCTIONS ###
#################


def draw_bold_text(
    frame,
    text,
    position,
    font_scale=2.5,
    color=(255, 255, 255),
    thickness=5,
    line_type=cv2.LINE_AA,
    offset=2,
):
    """
    Draws bolder text with slight offsets to create a thicker appearance.

    Args:
        frame (numpy.ndarray): The image/frame on which to draw the text.
        text (str): The text to be drawn.
        position (tuple): (x, y) position of the text.
        font_scale (float): Font scale.
        color (tuple): Text color (BGR format).
        thickness (int): Thickness of the text.
        line_type: Line type for drawing text.
        offset (int): Offset for creating bolder appearance.

    Returns:
        None (action performed directly on frame).
    """
    for offset_x, offset_y in [
        (-offset, -offset),
        (-offset, offset),
        (offset, -offset),
        (offset, offset),
    ]:
        offset_position = (position[0] + offset_x, position[1] + offset_y)
        cv2.putText(
            frame,
            text,
            offset_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            line_type,
        )


def initialize_pose_model(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.3
):
    """
    Initializes the MediaPipe's Pose Landmark Detection with given parameters.

    Args:
        Check parameters in documentation here: https://chuoling.github.io/mediapipe/solutions/pose.html

    Returns:
        mediapipe.solutions.pose.Pose: Initialized Pose Landmark Detection instance that is used for further processing in function 'process_frame'.
    """
    return mp_pose.Pose(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def process_frame(frame, pose):
    """
    Processes frame from videostream using the specified MediaPipe Pose model.

    Args:
        frame (numpy.ndarray): The input frame in BGR format.
        pose (mediapipe.solutions.pose.Pose): Initialized Pose model.

    Returns:
        Tuple[Any, numpy.ndarray]: A tuple containing
            [0]: the results from pose processing used for further processing in function 'extract_landmarks';
            [1]: the annotated frame (landmarks and landmark conections shown in videostream)
    """

    # Convert the frame from BGR to RGB color space as required by the Pose model.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)

    # Convert frame back to BGR color scheme.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw landmark and landmark connections on frame using given parameters for color/style (can be adjusted)
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=4, circle_radius=5  # white
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 0, 0), thickness=15, circle_radius=5  # blue
        ),
    )
    return results, frame


def extract_landmarks(results):
    """
    This function takes the results of pose processing and extracts specific pose landmarks
    (landmark indices 11 to 16 and 23 to 28) if available. The extracted coordinates are returned as
    a NumPy array. If no landmarks are available in the results, a zero-filled array of shape (12, 2)
    is returned.

    Args:
        results: The results object containing pose landmarks (obtained via function 'process_frame').

    Returns:
        numpy.ndarray: An array of extracted and processed landmark coordinates.
    """
    landmark_list = results.pose_landmarks

    if landmark_list:
        coordinate_list = []

        # Excluding facial, finger and feet landmarks
        filtered_landmarks = (
            landmark_list.landmark[11:17] + landmark_list.landmark[23:29]
        )

        # Extracting x and y coordinates only (excluding z coordinate, visibilty value etc.)
        for landmark in filtered_landmarks:
            x = landmark.x
            y_ = landmark.y
            coordinate_list.append([x, y_])
        coordinates = np.array(coordinate_list)

    # Fills an array with zeros when no landmarks are detected (shape: (12,2) -> 12 landmarks and their x,y coordinates)
    else:
        coordinates = np.zeros((12, 2))

    return coordinates


# This function for assigning labels to predicted poses was eventually replaced with the function 'predict_pose' to accomodate for prediction limitations.
# So, it's not used within play.py but I wanted to keep this to show the project's development progress ;)
def predict_pose_v1(landmark_coordinates, label_mapping, model):
    """
    Predicts a pose label based on given landmark coordinates and a trained model.

    This function assigns pose labels based on prediction confidence.
    It reassigns 'lazy poses' predicted with low confidence to the 'other' classification ('Pose'),
    aiming to enhance the game's challenge and robustness.

    Args:
        landmark_coordinates (numpy.ndarray): An array of shape (12, 2) containing pose landmark x,y coordinates obtained from the 'extract_landmarks' function.
        label_mapping (Dict[str, int]): A dictionary mapping pose labels to integer codes.
        model (Any): A trained model for predicting poses based on landmark coordinates.

    Returns:
        str: The predicted pose label, possibly reassigned to 'Pose' for low-confidence predictions.
    """
    # Flatten landmark coordinates array to match model input shape
    landmark_coordinates_flattened = landmark_coordinates.reshape(1, -1)

    # Get the prediction from the model
    prediction = model.predict(landmark_coordinates_flattened)

    # Find the index of the predicted pose
    predicted_pose_index = np.argmax(prediction)

    # Extract prediction confidence (probability)
    pred_prob = prediction[0][predicted_pose_index]

    # Match the index to the corresponding string label using the label mapping dictionary.
    # Defaulting to "Pose" if no match is found (unlikely).
    for label, index in label_mapping.items():
        if index == predicted_pose_index:
            predicted_label = label

    # Reassign to "Pose" class if prediction confidence is below 0.7
    if pred_prob < CONFIDENCE_THRESHOLD:
        predicted_label = "Pose"

    return predicted_label


def predict_pose_v2(landmark_coordinates, label_mapping, model):
    """
    Predicts a pose label based on given landmark coordinates and a trained model.

    This function assigns pose labels based on prediction confidence and addresses
    certain limitations of the model's behavior (see 'Prediction Probablities' in train_model.ipynb).

    Args:
        landmark_coordinates (numpy.ndarray): An array of shape (12, 2) containing pose landmark x,y coordinates obtained from the 'extract_landmarks' function.
        label_mapping (Dict[str, int]): A dictionary mapping pose labels to integer codes.
        model (Any): A trained model for predicting poses based on landmark coordinates.

    Returns:
        str: The predicted pose label, possibly reassigned to 'Pose' for low-confidence predictions.
    """
    # Flatten landmark coordinates array to match model input shape
    landmark_coordinates_flattened = landmark_coordinates.reshape(1, -1)

    # Get the prediction from the model
    prediction = model.predict(landmark_coordinates_flattened)

    # Find the index of the most likely predicted pose
    predicted_pose_index = np.argmax(prediction)

    # Find the index of the second most likely predicted pose
    predicted_pose_index_2 = np.argsort(prediction[0])[-2]

    # Extract prediction confidence (probability)
    pred_prob = prediction[0][predicted_pose_index]

    # Match the indices to the corresponding string labels using the label mapping dictionary.
    for label, index in label_mapping.items():
        if index == predicted_pose_index:
            predicted_label = label
        if index == predicted_pose_index_2:
            predicted_label_2 = label

    # Confirms predicted pose 'Stand' when confidence threshold of 0.5 is met and 'Stand' is the most likely prediction.
    if predicted_label == "Stand" and pred_prob > CONFIDENCE_THRESHOLD_STAND:
        predicted_label = "Stand"
    # Reassigns pose label to 'Stand' if the most likely prediction is 'Pose' with 'Stand' being the next most likely (with at min. confidence of 0.2).
    elif (
        predicted_label == "Pose"
        and predicted_label_2 == "Stand"
        and prediction[0][np.argsort(prediction[0])[-2]]
        > CONFIDENCE_THRESHOLD_POSE_TO_STAND
    ):
        predicted_label = "Stand"
    # Reassigns to "Pose" class if prediction confidence is below 0.7 and predicted pose is not 'Stand'.
    elif pred_prob < CONFIDENCE_THRESHOLD and predicted_label != "Stand":
        predicted_label = "Pose"

    return predicted_label


def save_data(SAVE_PATH, current_pose, frame, results, file_substring="_"):
    """
    Saves annotated frames (obtained via function 'process_frame')
    and extracted & processed landmark coordinates (obtained via functions 'process_frame' and 'extract_landmarks')
    to specified path.

    Args:
        SAVE_PATH (str): Path to folder in which data is to be saved.
        current_pose (str): Name of the pose for which data is saved, it is used to create subfolders within data collection folder.
        frame (numpy.ndarray): Annotated frame to be saved.
        results: Result object obtained via function 'process_frame'.
        file_substring (str): Optional substring for the filename; useful for recording 'to be predicted' and 'actually predicted' data when executing play.py.

    Returns:
        None.
    """
    # Define folder paths for saving frames and landmarks
    picture_folder = f"{SAVE_PATH}/frames/{current_pose}"
    landmarks_folder = f"{SAVE_PATH}/landmarks/{current_pose}"

    # Creates folders for saving if they do not exist.
    for folder in [picture_folder, landmarks_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Generates unique and chronological timestamps for filenames.
    timestamp = int(time.time())

    # Extracts and processes landmark coordinates from results of frame processing.
    landmark_coordinates = extract_landmarks(results)

    # Define filenames for the saved frames and landmarks.
    filename_pictures = (
        f"{picture_folder}/{current_pose}_{file_substring}_{timestamp}.jpg"
    )
    filename_landmarks = (
        f"{landmarks_folder}/{current_pose}_{file_substring}_{timestamp}.npy"
    )

    # Saves frame (jpg) and landmark coordinates (numpy array) to specified paths.
    cv2.imwrite(filename_pictures, frame)
    np.save(filename_landmarks, landmark_coordinates)


def display_instructions(frame):
    """
    Displays instructions on how to start data collection.

    Args:
        frame (numpy.ndarray): The image/frame on which to draw the text.

    Returns:
        None (action performed directly on frame).
    """
    common_text_params = {
        "font_scale": 2,
        "color": (0, 255, 255),  # yellow
        "thickness": 2,
        "offset": 1,
    }

    text_lines = ["Press SPACE to start countdown", "Press Q to quit"]

    for i, text in enumerate(text_lines):
        text_position = (30, 100 + i * 70)
        draw_bold_text(frame, text, text_position, **common_text_params)


def blink_screen(
    frame, color_channel, record_time, blink_flag, alpha=0.5, blink_duration=0.2
):
    """
    Applies a screen blink effect to the frame.

    Args:
        frame (numpy.ndarray): Input image frame.
        color_channel (int): BGR color channel to modify ([0]: blue, [1]: green, etc.)
        alpha (float): Opacity of the blink effect (default is 0.5).
        record_time (float): The time at which the blink effect started (= time of pose recording/classification).
        blink_duration (float): The duration of the blink effect (default is 0.2 seconds).
        blink_flag (bool): Flag to control when the blink effect should be applied.

    Returns:
        numpy.ndarray: The modified image frame (frame merged with the blink effect.)
        bool: Updated blink_flag.
    """
    blink_rect = np.zeros_like(
        frame
    )  # Initializes blinking array of the same shape/size as the frame.
    blink_rect[:, :, color_channel] = 255  # Sets blinking array the specified color.
    frame = cv2.addWeighted(
        frame, 1 - alpha, blink_rect, alpha, 0
    )  # Merges frame and blinking array.

    # Turns blinking off after blink_duration.
    if time.time() - record_time >= blink_duration:
        blink_flag = False

    return frame, blink_flag


def display_gameover_message(frame, points, ROUNDS):
    """
    Displays the restart message on the frame when the game is over.

    Args:
        frame (numpy.ndarray): The frame on which to display the message.
        points (int): The player's score.
        ROUNDS (int): The total number of rounds in the game.

    Returns:
        None
    """
    # Shows game over message
    game_over_text = "GAME OVER!"
    game_over_position = (350, 100)
    draw_bold_text(
        frame,
        game_over_text,
        game_over_position,
        font_scale=3,
        color=(0, 0, 255),  # red
        thickness=3,
    )

    points_text = f"Points: {points}/{ROUNDS}"
    points_position = (350, 250)
    draw_bold_text(
        frame,
        points_text,
        points_position,
        font_scale=3,
        color=(255, 0, 0),  # blue
        thickness=3,
    )

    # Show restart and exit instructions in a blinking manner.
    if int(time.time() * 2) % 2 == 0:
        restart_text = "Restart: 'SPACE'"
        restart_position = (250, 560)

        draw_bold_text(
            frame,
            restart_text,
            restart_position,
            font_scale=3,
            color=(0, 0, 255),  # red
            thickness=3,
        )

        exit_text = "Exit: 'Q'"
        exit_position = (450, 660)

        draw_bold_text(
            frame,
            exit_text,
            exit_position,
            font_scale=3,
            color=(0, 0, 255),  # red
            thickness=3,
        )
