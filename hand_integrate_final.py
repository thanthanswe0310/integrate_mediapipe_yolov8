import csv
import copy
import argparse
import itertools
from collections import Counter, deque
import cv2 as cv
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier


def load_image_paths(image_dir):
    """ Load image paths from a file or directory. """
    if os.path.isfile(image_dir):
        return [image_dir]
    elif os.path.isdir(image_dir):
        return [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        raise FileNotFoundError(f"The specified path {image_dir} does not exist.")


def get_args():
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("--width", help='Image width', type=int, default=960)
    parser.add_argument("--height", help='Image height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true', help="Use static image mode")
    parser.add_argument("--min_detection_confidence", help='Minimum detection confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='Minimum tracking confidence', type=float, default=0.5)
    parser.add_argument("--image_dir", help='Path to the image file or directory containing images', type=str, default='inference/images')
    parser.add_argument('--view-img', action='store_true', help='Display results')
    parser.add_argument('--save-txt', action='store_true', help='Save results to *.txt')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    image_dir = args.image_dir

    print("Image File: ", image_dir)
    
    # Check if the provided path is a file or directory
    image_paths = load_image_paths(image_dir)

    # Load YOLOv8 model
    model = YOLO('/path/to/yolov8_best.pt')

    # Mediapipe Hands load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

    # FPS Measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0

    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        image = cv.imread(image_path)

        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue
        image = cv.resize(image, (cap_width, cap_height))
        debug_image = copy.deepcopy(image)

        # Object Detection with YOLOv8
        yolo_results = model(image)[0]
        for result in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = result.conf
            class_id = int(result.cls)
            cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(debug_image, f'{model.names[class_id]} {conf.item():.2f}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hand Detection with Mediapipe
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                logging_csv(-1, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id], point_history_classifier_labels[most_common_fg_id[0][0]])
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        fps = cvFpsCalc.get()
        debug_image = draw_info(debug_image, fps, mode, -1)

        # Save image if needed
        output_image_path = os.path.join(image_dir, f"processed_{os.path.basename(image_path)}") if os.path.isdir(image_dir) else f"processed_{os.path.basename(image_path)}"
        cv.imwrite(output_image_path, debug_image)
        print(f"Processed image saved: {output_image_path}")

        # Optionally display the image
        if args.view_img:
            cv.imshow('Hand Gesture Recognition', debug_image)
            cv.waitKey(0)  # Wait for a key press to move to the next image

    cv.destroyAllWindows()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Normalizing coordinates
    for index, landmark in enumerate(temp_landmark_list):
        temp_landmark_list[index] = [(landmark[0] - landmark_list[0][0]) / (landmark_list[8][0] - landmark_list[0][0]),
                                      (landmark[1] - landmark_list[0][1]) / (landmark_list[8][1] - landmark_list[0][1])]

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    temp_point_history = copy.deepcopy(point_history)
    point_history_array = np.array(temp_point_history).flatten()

    if len(point_history_array) == 0:
        return []

    return point_history_array.reshape(-1, 2).tolist()


def draw_landmarks(image, landmark_list):
    for point in landmark_list:
        cv.circle(image, (point[0], point[1]), 5, (0, 255, 0), -1)
    return image


def draw_info_text(image, brect, handedness, sign_label, gesture_label):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    cv.putText(image, f'Hand: {handedness.classification[0].label}', (brect[0], brect[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv.putText(image, f'Sign: {sign_label}', (brect[0], brect[1] - 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv.putText(image, f'Gesture: {gesture_label}', (brect[0], brect[1] - 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def draw_point_history(image, point_history):
    for point in point_history:
        cv.circle(image, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, f'FPS: {fps}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(image, f'Mode: {mode}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(image, f'Number: {number}', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image


def logging_csv(number, mode, landmark_list, point_history_list):
    with open('log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([number, mode] + list(itertools.chain(*landmark_list)) + list(itertools.chain(*point_history_list)))


if __name__ == "__main__":
    main()
