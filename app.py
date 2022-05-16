#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import pynput
import matplotlib.pyplot as plt
import subprocess


from collections import Counter
from collections import deque
from pynput.keyboard import Key , Controller

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier


def main():
    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode='store_true',
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    keypoint_classifier = KeyPointClassifier()



    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    winname = "Hand gesture detect"
    cv.namedWindow(winname) 
    cv.moveWindow(winname, 0,0)   
    # Finger gesture history ################################################

    #  ########################################################################
    mode = 0
    output=3
    keyborard=Controller()
    img_counter = 0
    try:
        file = open("LastNumber.txt")
        img_counter=int(file.read())
    except IOError:
        print("File not accessible")

    p = subprocess.Popen("juego\Proyecto_Pollo.exe")

    while True:
        fps = cvFpsCalc.get()
        
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            f = open("LastNumber.txt", "w")
            f.write(str(img_counter))
            f.close()
            file.close()
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        
        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(
                    debug_image, 
                    hand_landmarks)

                # Landmark calculation
                landmark_list = calc_landmark_list(
                    debug_image, 
                    hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                # Write to the dataset file
                img_counter=logging_csv(
                    image,
                    img_counter,
                    number, 
                    mode, 
                    pre_processed_landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(
                    pre_processed_landmark_list)

                output = output_Signal(
                    output, 
                    keyborard, 
                    hand_sign_id)

                # Drawing part
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                    )

                debug_image=draw_bounding_rect(
                    use_brect, 
                    debug_image, 
                    brect)

                debug_image=draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                )

        debug_image=draw_info(
            debug_image, 
            fps, 
            mode, 
            number)

        # Screen reflection #############################################################

        cv.imshow(winname, debug_image)
    p.terminate()
    cap.release()
    cv.destroyAllWindows()

def output_Signal(output, keyborard, hand_sign_id):
    if output != hand_sign_id:
        output = hand_sign_id
                    
        if hand_sign_id==0:
                        #pynput command jump
            print("Jump!")
            keyborard.press('w')
            keyborard.release('w')
            
        if hand_sign_id==1:
                        #go right
            print("Right!")
            keyborard.press('d')
            keyborard.release('d')

        if hand_sign_id ==2:
                        #goleft
            print("Left!")
            keyborard.press('a')
            keyborard.release('a')
        if hand_sign_id ==4:
                        #reset
            print("Reset!")
            keyborard.press('q')
            keyborard.release('q')
        if hand_sign_id ==5:
                        #back
            print("Back!")
            keyborard.press('s')
            keyborard.release('s')
    return output


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
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
    plt.imshow(image)

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(image,img_counter,number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        image_path="model/keypoint_classifier/image/{}_{}.png".format(number,img_counter)
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        cv.imwrite(image_path, cv.cvtColor(image, cv.COLOR_BGR2RGB))
        print("{} written!".format(image_path))
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
        img_counter += 1
    return img_counter

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 255), 3)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 255), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv.LINE_AA)

    return image


def draw_info(image, fps, mode, number):
    
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point']
    if mode == 1:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()
