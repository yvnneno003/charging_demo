#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   Copyright (C) 2023 MSI-FUNTORO
#
#   Licensed under the MSI-FUNTORO License, Version 1.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.funtoro.com/global/
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
License Plate Recognition
'''
import os
import sys
import time
import traceback
import cv2
import numpy as np
from tqdm import tqdm
from colorama import init, Fore

from yolov5.detect.detect import Detect

import random
import time
from paho.mqtt import client as mqtt_client

# initialize colorama
init(autoreset=True)

##########  Python Library Version ##########
recommend_package_dict = {'torch': '==1.8.0',
                          'onnxruntime': '==1.14.1',
                          'onnxruntime-gpu': '==1.14.1',
                          'tensorflow-cpu': '==2.11.0',
                          'tensorflow-gpu': '==2.11.0',
                          'tensorrt': '>=8.6.1',
                          'opencv-python': '>=4.5.2.52',
                          'colorama': '>=0.4.5'}
#############################################


class LicensePlateRecognition:
    # Vehicle license plate detection
    _vehicle_license_plate_detection_model_path = None
    _vehicle_license_plate_detection_model = None
    _vehicle_license_plate_classes_txt_path = 'classes/classes.txt'
    _vehicle_license_plate_classes = []
    _vehicle_confidence_threshold = 0.5
    _vehicle_iou_threshold = 0.4
    _license_plate_confidence_threshold = 0.5
    _license_plate_iou_threshold = 0.4

    # License plate text detection
    _license_plate_text_detection_model_path = None
    _license_plate_text_detection_model = None
    _license_plate_text_classes_txt_path = 'classes/classes-license-plate-text.txt'
    _license_plate_text_classes = []
    _license_plate_text_confidence_threshold = 0.5
    _license_plate_text_iou_threshold = 0.4

    # Warm up
    _warm_up_image_path = 'images/warm_up_image.jpg'

    # Debug
    _debug = False

    # Result Dictionary Key
    RESULT_LABEL_KEY = Detect.result_label_key
    RESULT_BOX_KEY = Detect.result_box_key
    RESULT_CONFIDENCE_KEY = Detect.result_confidence_key
    RESULT_VEHICLE_KEY = 'vehicle'
    RESULT_LICENSE_PLATE_KEY = 'license plate'
    RESULT_LICENSE_PLATE_OVERLAY_PERCENT_KEY = 'overlay percent'
    RESULT_LICENSE_PLATE_TEXT_KEY = 'license plate text'
    RESULT_LICENSE_PLATE_IMAGE_KEY = 'license plate image'


    def __init__(self, vehicle_license_plate_detection_model_path, license_plate_text_detection_model_path, warm_up_image_path,
                 vehicle_license_plate_classes_txt_path, vehicle_confidence_threshold, vehicle_iou_threshold, license_plate_confidence_threshold, license_plate_iou_threshold,
                 license_plate_text_classes_txt_path, license_plate_text_confidence_threshold, license_plate_text_iou_threshold,
                 debug, show_init=True):
        self._vehicle_license_plate_detection_model_path = vehicle_license_plate_detection_model_path
        self._license_plate_text_detection_model_path = license_plate_text_detection_model_path
        self._warm_up_image_path = warm_up_image_path
        self._vehicle_license_plate_classes_txt_path = vehicle_license_plate_classes_txt_path
        self._vehicle_confidence_threshold = vehicle_confidence_threshold
        self._vehicle_iou_threshold = vehicle_iou_threshold
        self._license_plate_confidence_threshold = license_plate_confidence_threshold
        self._license_plate_iou_threshold = license_plate_iou_threshold
        self._license_plate_text_classes_txt_path = license_plate_text_classes_txt_path
        self._license_plate_text_confidence_threshold = license_plate_text_confidence_threshold
        self._license_plate_text_iou_threshold = license_plate_text_iou_threshold
        self._debug = debug
        if show_init:
            print(Fore.GREEN + '\n==========   License Plate Recognition    ==========')
            print(Fore.GREEN + 'vehicle license plate detection model path: ', self._vehicle_license_plate_detection_model_path)
            print(Fore.GREEN + 'warm up image pathL ', self._warm_up_image_path)
            print(Fore.GREEN + 'vehicle license plate classes txt path: ', self._vehicle_license_plate_classes_txt_path)
            print(Fore.GREEN + 'vehicle confidence threshold: ', self._vehicle_confidence_threshold)
            print(Fore.GREEN + 'vehicle iou threshold: ', self._vehicle_iou_threshold)
            print(Fore.GREEN + 'license plate confidence threshold: ', self._license_plate_confidence_threshold)
            print(Fore.GREEN + 'license plate iou threshold: ', self._license_plate_iou_threshold)
            print(Fore.GREEN + 'license plate text detection model path: ', self._license_plate_text_detection_model_path)
            print(Fore.GREEN + 'license plate text classes txt path: ', self._license_plate_text_classes_txt_path)
            print(Fore.GREEN + 'license plate text confidence threshold: ', self._license_plate_text_confidence_threshold)
            print(Fore.GREEN + 'license plate text iou threshold: ', self._license_plate_text_iou_threshold)
            print(Fore.GREEN + 'debug: ', self._debug)

        self._init_vehicle_license_plate_detection_model()
        self._init_license_plate_text_detection_model()
        self._init_license_plate_classes()

        if show_init:
            print(Fore.GREEN + '====================================================\n')


    def _init_vehicle_license_plate_detection_model(self):
        print(Fore.GREEN + '\nInitialize vehicle license plate detection model')
        self._vehicle_license_plate_detection_model = Detect(model_path=self._vehicle_license_plate_detection_model_path,
                                                             warm_up_image_path=self._warm_up_image_path,
                                                             confidence_threshold=min(self._vehicle_confidence_threshold, self._license_plate_confidence_threshold),
                                                             iou_threshold=min(self._vehicle_iou_threshold, self._license_plate_iou_threshold),
                                                             debug=False,
                                                             show_init=False)
        if not self._vehicle_license_plate_detection_model.is_initialize():
            self._vehicle_license_plate_detection_model = None


    def _init_license_plate_text_detection_model(self):
        print(Fore.GREEN + '\nInitialize license plate text detection model')
        self._license_plate_text_detection_model = Detect(model_path=self._license_plate_text_detection_model_path,
                                                          warm_up_image_path=self._warm_up_image_path,
                                                          confidence_threshold=self._license_plate_text_confidence_threshold,
                                                          iou_threshold=self._license_plate_text_iou_threshold,
                                                          debug=False,
                                                          show_init=False)
        if not self._license_plate_text_detection_model.is_initialize():
            self._license_plate_text_detection_model = None


    def _init_license_plate_classes(self):
        with open(self._license_plate_text_classes_txt_path, 'r') as file:
            for line in file.readlines():
                self._license_plate_text_classes.append(line.replace('\n', ''))


    def recognition(self, image=None, image_path=None):
        '''
        Recognition

        Args:
            image (numpy): The BGR image you want to use for detection.
            image_path (str): The image path you want to use for detection.

        Returns: list containing all the vehicle information detected with their
                    vehicle [RESULT_VEHICLE_KEY] (dict):
                        label index [RESULT_LABEL_KEY] (int): Index of label
                        box [RESULT_BOX_KEY] (list[float]): [top x, top y, button x, bottom y]
                        confidence [RESULT_CONFIDENCE_KEY] (float): 0 ~ 1

                    license plate [RESULT_LICENSE_PLATE_KEY] (dict):
                        label index [RESULT_LABEL_KEY] (int): Index of label
                        box [RESULT_BOX_KEY] (list[float]): [top x, top y, button x, bottom y]
                        confidence [RESULT_CONFIDENCE_KEY] (float): 0 ~ 1
                        overlay percent [RESULT_LICENSE_PLATE_OVERLAY_PERCENT_KEY] (float): Percent of vehicle and license plate overlay, range: 0 ~1
                        license plate text [RESULT_LICENSE_PLATE_TEXT_KEY] (str): License plate text
                        license plate image [RESULT_LICENSE_PLATE_IMAGE_KEY] (numpy): License plate image
        '''
        try:
            success = False
            start_time = time.time()
            if self._debug:
                print('\n********** Recognition **********')

            if self._vehicle_license_plate_detection_model is None:
                raise Exception('Vehicle license plate detection model is not initialize.')
            if self._license_plate_text_detection_model is None:
                raise Exception('License plate text detection model is not initialize.')

            if image is None:
                if image_path is None:
                    raise Exception('Image and image path is None.')
                if not os.path.exists(image_path):
                    raise Exception('Image (' + str(image_path) + ') is not exist.')

                # load image
                image = cv2.imread(image_path)

            # detect vehicle and license plate
            vehicle_list = self._detect_vehicle_license_plate(image=image)

            for vehicle in vehicle_list:
                if vehicle[self.RESULT_LICENSE_PLATE_KEY] is not None:
                    license_plate_box = np.array(vehicle[self.RESULT_LICENSE_PLATE_KEY][Detect.result_box_key], dtype=np.int32)

                    # crop license plate
                    # license_plate_image = image[license_plate_box[1]-5:license_plate_box[3]+5, license_plate_box[0]-5:license_plate_box[2]+5]
                    license_plate_image = self._crop_license_plate(image=image, license_plate_box=license_plate_box)

                    # blur_img = cv2.GaussianBlur(license_plate_image, (0, 0), 100)
                    # license_plate_image = cv2.addWeighted(license_plate_image, 1.5, blur_img, -0.5, 0)

                    # license plate recognition
                    license_plate = self._license_plate_recognition(image=license_plate_image)
                    vehicle[self.RESULT_LICENSE_PLATE_KEY][self.RESULT_BOX_KEY] = license_plate_box
                    vehicle[self.RESULT_LICENSE_PLATE_KEY][self.RESULT_LICENSE_PLATE_TEXT_KEY] = license_plate
                    vehicle[self.RESULT_LICENSE_PLATE_KEY][self.RESULT_LICENSE_PLATE_IMAGE_KEY] = license_plate_image
            success = True

        except Exception as ex:
            print(Fore.RED + 'Recognition error. ' + str(ex) + '\n' + str(traceback.format_exc()))
            vehicle_list = []

        finally:
            cost_time = time.time() - start_time
            fps = 1 / cost_time
            if self._debug:
                if success:
                    print('Vehicle number: ' + str(len(vehicle_list)))
                    for vehicle in vehicle_list:
                        debug_str = '{\'' + self.RESULT_VEHICLE_KEY + '\': ' + str(vehicle[self.RESULT_VEHICLE_KEY]) + \
                                    ', \'' + self.RESULT_LICENSE_PLATE_KEY + '\': '
                        license_plate = vehicle[self.RESULT_LICENSE_PLATE_KEY]
                        if license_plate is not None:
                            debug_str += '{\'' + self.RESULT_LABEL_KEY + '\': ' + str(license_plate[self.RESULT_LABEL_KEY]) + \
                                         ', \'' + self.RESULT_BOX_KEY + '\': ' + str(license_plate[self.RESULT_BOX_KEY]) + \
                                         ', \'' + self.RESULT_CONFIDENCE_KEY + '\': ' + str(license_plate[self.RESULT_CONFIDENCE_KEY]) + \
                                         ', \'' + self.RESULT_LICENSE_PLATE_OVERLAY_PERCENT_KEY + '\': ' + str(license_plate[self.RESULT_LICENSE_PLATE_OVERLAY_PERCENT_KEY]) + \
                                         ', \'' + self.RESULT_LICENSE_PLATE_TEXT_KEY + '\': ' + str(license_plate[self.RESULT_LICENSE_PLATE_TEXT_KEY]) + \
                                         ', \'' + self.RESULT_LICENSE_PLATE_IMAGE_KEY + '\': ' + str(license_plate[self.RESULT_LICENSE_PLATE_IMAGE_KEY].shape) + '}'
                        else:
                            debug_str += 'None}'
                        print(debug_str)
                    print('Cost time: ' + str(cost_time) + 's. FPS: ' + str(fps))
                print('*********************************\n')
            return vehicle_list, fps


    def recognition_source(self, source, output_folder_path=None, show=True, show_fps=False):
        print(Fore.YELLOW + '\n**********   Recognition Source   **********')
        print(Fore.YELLOW + 'source: ', source)
        print(Fore.YELLOW + 'output folder path: ', output_folder_path)
        print(Fore.YELLOW + 'show: ', str(show))
        print(Fore.YELLOW + 'show fps: ', str(show_fps))

        try:
            if self._vehicle_license_plate_detection_model is None:
                raise Exception('Recognition source failed. Vehicle license plate detection model is not initialize!!')
            if self._license_plate_text_detection_model is None:
                raise Exception('Recognition source failed. License plate text detection model is not initialize!!')

            # get source list
            recognition_file_list = []
            is_camera = False
            if os.path.isfile(source):
                recognition_file_list.append(source)
            elif os.path.isdir(source):
                for file_name in os.listdir(source):
                    recognition_file_list.append(os.path.join(source, file_name))
            else:
                is_camera = True
            print(Fore.YELLOW + 'Source: ', 'Camera ' + str(source) if is_camera else recognition_file_list)

            # create save result folder
            if output_folder_path is not None:
                # create output folder
                if not os.path.exists(output_folder_path):
                    os.mkdir(output_folder_path)

            # recognition
            if is_camera:
                self._recognition_video_source(video_path=int(args.source), is_camera=is_camera, output_folder_path=output_folder_path, show=True, show_fps=show_fps)
            else:
                for recognition_file in recognition_file_list:
                    _, file_extension = os.path.splitext(recognition_file)

                    if file_extension in support_image_format:
                        ''' Image '''
                        _ = self._recognition_image_source(image_path=recognition_file, output_folder_path=output_folder_path, show=show, show_fps=show_fps)

                    elif file_extension in support_video_format:
                        ''' Video '''
                        self._recognition_video_source(video_path=recognition_file, output_folder_path=output_folder_path, show=show, show_fps=show_fps)

        except Exception as ex:
            print(Fore.RED + 'Recognition source error. ' + str(ex) + '\n' + str(traceback.format_exc()))

        finally:
            print(Fore.YELLOW + '********************************************\n')


    def _detect_vehicle_license_plate(self, image):
        try:
            vehicle_list = []
            license_plate_list = []

            results, _, _ = self._vehicle_license_plate_detection_model.detect(image=image)
            for result in results:
                label = result[Detect.result_label_key]
                if label == 2 or label == 3 or label == 4 or label == 6 or label == 8:
                    vehicle_list.append({self.RESULT_VEHICLE_KEY: result,
                                         self.RESULT_LICENSE_PLATE_KEY: None})
                elif label == 9:
                    license_plate_list.append(result)

            for license_plate in license_plate_list:
                best_match_vehicle_index = None
                best_license_plate_overlap_percent = 0
                for index, vehicle in enumerate(vehicle_list):
                    if vehicle[self.RESULT_LICENSE_PLATE_KEY] is not None:
                        continue

                    # calculate license plate overlap percent with vehicle
                    license_plate_overlap_percent = self._calculate_overlap_percent(box1=license_plate[Detect.result_box_key],
                                                                                    box2=vehicle[self.RESULT_VEHICLE_KEY][Detect.result_box_key])

                    if license_plate_overlap_percent > 0.8:
                        if best_match_vehicle_index is None:
                            best_match_vehicle_index = index
                            best_license_plate_overlap_percent = license_plate_overlap_percent

                        elif license_plate_overlap_percent > best_license_plate_overlap_percent:
                            best_match_vehicle_index = index
                            best_license_plate_overlap_percent = license_plate_overlap_percent

                if best_match_vehicle_index is not None:
                    license_plate[self.RESULT_LICENSE_PLATE_OVERLAY_PERCENT_KEY] = best_license_plate_overlap_percent
                    vehicle_list[best_match_vehicle_index][self.RESULT_LICENSE_PLATE_KEY] = license_plate


        except Exception as ex:
            print(Fore.RED + 'Detect vehicle and license plate error. ' + str(ex) + '\n' + str(traceback.format_exc()))
            vehicle_list = []

        finally:
            return vehicle_list


    def _license_plate_recognition(self, image):
        try:
            license_plate = ''
            results, _, _ = self._license_plate_text_detection_model.detect(image=image)
            results = sorted(results, key=lambda x:x[Detect.result_box_key][0])
            for result in results:
                license_plate += str(self._license_plate_text_classes[int(result[Detect.result_label_key])])

        except Exception as ex:
            print(Fore.RED + 'License plate recognition error. ' + str(ex) + '\n' + str(traceback.format_exc()))

        finally:
            return license_plate


    def _calculate_overlap_percent(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x1 >= x2 or y1 >= y2:
            overlap = 0
        else:
            overlap = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

        return overlap / box1_area


    def _crop_license_plate(self, image, license_plate_box):
        expansion_size_percent = 10
        image_height, image_width, _ = image.shape
        top_x, top_y, bottom_x, bottom_y = license_plate_box[0], license_plate_box[1], license_plate_box[2], license_plate_box[3]
        width = bottom_x - top_x
        height = bottom_y - top_y

        top_x -= int(width / expansion_size_percent)
        top_x = top_x if top_x >= 0 else 0
        bottom_x += int(width / expansion_size_percent)
        bottom_x = bottom_x if bottom_x <= image_width else image_width

        top_y -= int(height / expansion_size_percent)
        top_y = top_y if top_y >= 0 else 0
        bottom_y += int(height / expansion_size_percent)
        bottom_y = bottom_y if bottom_y <= image_height else image_height

        license_plate_image = image[top_y:bottom_y, top_x:bottom_x]
        return license_plate_image


    def _recognition_image_source(self, image_path=None, bgr_image=None, output_folder_path=None, show=True, show_fps=False):
        if bgr_image is None:
            image = cv2.imread(image_path)
        else:
            image = bgr_image.copy()

        # predict
        vehicle_list, fps = self.recognition(image=image)
        # print(vehicle_list)

        # color
        detected_license_plate_color = (0, 128, 0)
        undetected_license_plate_color = (0, 0, 255)

        for vehicle in vehicle_list:
            vehicle_box = vehicle[LicensePlateRecognition.RESULT_VEHICLE_KEY][LicensePlateRecognition.RESULT_BOX_KEY].astype(int)

            license_plate = vehicle[LicensePlateRecognition.RESULT_LICENSE_PLATE_KEY]
            if license_plate is not None:
                vehicle_box_color = detected_license_plate_color
                license_plate_text = license_plate[LicensePlateRecognition.RESULT_LICENSE_PLATE_TEXT_KEY]
                print('license_plate_text='+license_plate_text)
                if license_plate_text != '':
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    text_size, _ = cv2.getTextSize(license_plate_text, font, font_scale, thickness)
                    cv2.rectangle(image, (vehicle_box[0], vehicle_box[1]), (vehicle_box[0] + text_size[0], vehicle_box[1] - text_size[1] - 5), vehicle_box_color, -1)
                    cv2.putText(image, license_plate_text, (vehicle_box[0], vehicle_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                    #publish(client,"user/lilywu@msi.com;"+license_plate_text+";unlock")
                    publish(client,"user/lilywu@msi.com;"+license_plate_text+";unlock")
                    
                # draw license plate
                license_plate_image = license_plate[LicensePlateRecognition.RESULT_LICENSE_PLATE_IMAGE_KEY]
                h, w, _ = license_plate_image.shape
                image[vehicle_box[1]:vehicle_box[1] + h, vehicle_box[0]:vehicle_box[0] + w] = license_plate_image

            else:
                vehicle_box_color = undetected_license_plate_color

            # draw vehicle box
            cv2.rectangle(image, (vehicle_box[0], vehicle_box[1]), (vehicle_box[2], vehicle_box[3]), vehicle_box_color, 2)

        # FPS
        if show_fps:
            cv2.putText(image, "FPS: " + format(fps, '.1f'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if output_folder_path is not None and image_path is not None:
            save_path = os.path.join(output_folder_path, os.path.splitext(os.path.basename(image_path))[0] + '.jpg')
            cv2.imwrite(save_path, image)

        if show:
            cv2.imshow('Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image, fps

    def _recognition_video_source(self, video_path, is_camera=False, output_folder_path=None, show=True, show_fps=False):
        if is_camera:
            from utils.opencv_camera_utils import OpenCVCameraUtils
            opencv_camera_utils = OpenCVCameraUtils()
            video = opencv_camera_utils.open(camera_id=video_path, width=1920, height=1080, fps=30, codec=1196444237.0)
        else:
            video = cv2.VideoCapture(video_path)
            video_total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = tqdm(total=video_total_frame_count)

        avg_fps = 0
        result_video = None
        if output_folder_path is not None:
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')

            if is_camera:
                save_path = os.path.join(output_folder_path, 'camera_' + str(video_path) + '.mp4')
            else:
                save_path = os.path.join(output_folder_path, os.path.splitext(os.path.basename(video_path))[0] + '.mp4')
            result_video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        while video.isOpened():
            if not is_camera:
                progress.set_description(video_path + ' [Avg fps: {:.0f}]'.format(avg_fps))
                progress.update()
            ret, frame = video.read()
            if not ret:
                print(Fore.RED + 'Can not receive frame (end frame). Exiting...')
                break

            # detect
            frame, fps = self._recognition_image_source(bgr_image=frame, show=False, show_fps=show_fps)
            if avg_fps == 0:
                avg_fps = fps
            else:
                avg_fps = (avg_fps + fps) / 2

            # save result frame to video
            if result_video is not None:
                result_video.write(frame)

            # show result image
            if show:
                cv2.imshow('Result', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        video.release()
        cv2.destroyAllWindows()
        if result_video is not None:
            result_video.release()


'''
=============================
Parse Default
=============================
'''
# vehicle and license plate detection
default_vehicle_license_plate_model_path = './models_zoo/yolov5/n/vehicle-license-plate-detection-n-alan-320-v1/vehicle-license-plate-detection-n-alan-320-v1.onnx'
default_warm_up_image_path = './images/warm_up_image.png'
default_vehicle_license_plate_classes_txt_path = './classes/classes.txt'
default_vehicle_confidence_threshold = 0.5
default_vehicle_iou_threshold = 0.4
default_license_plate_confidence_threshold = 0.5
default_license_plate_iou_threshold = 0.4
default_license_plate_overlap_percent_threshold = 0.8

# license plate text detection
default_license_plate_text_model_path = './models_zoo/yolov5/n/license-plate-text-detection-n-alan-128-v1/license-plate-text-detection-n-alan-128-v1.onnx'
default_license_plate_text_classes_txt_path = './classes/classes-license-plate-text.txt'
default_license_plate_text_confidence_threshold = 0.5
default_license_plate_text_iou_threshold = 0.4

# source
default_detect_source_path = './images/test_license_plate_recognition.png'
# default_detect_source_path = './videos/video_04.mp4'
support_image_format = ['.jpg', '.jpeg', '.png', '.bmp', '.dib', '.webp', '.sr', '.ras', '.tiff', '.tif']
support_video_format = ['.mp4', '.avi']

# output
default_output_folder_path = './outputs'
default_show = False
default_show_fps = False

# debug
default_debug = False

# mqtt
serverIp = '60.251.140.228'
port = 1883
topic ="device/5760/charger/TW*MSI*E000151"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'

'''
=============================
Args Parse
=============================
'''
def parse():
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(description='License Plate Recognition ONNX', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-vmp', '--vehicle_license_plate_model_path', type=str, default=default_vehicle_license_plate_model_path, help='Path of vehicle and license plate detection model.' +
                                                                                                                                       '\n\tDefault: ' + str(default_vehicle_license_plate_model_path))
    parser.add_argument('-lmp', '--license_plate_text_model_path', type=str, default=default_license_plate_text_model_path, help='Path of license plate text detection model.' +
                                                                                                                                 '\n\tDefault: ' + str(default_license_plate_text_model_path))
    parser.add_argument('-wip', '--warm_up_image_path', type=str, default=default_warm_up_image_path, help='Path of warm up image.' +
                                                                                                           '\n\tDefault: ' + str(default_warm_up_image_path))
    parser.add_argument('-vcp', '--vehicle_license_plate_classes_txt_path', type=str, default=default_vehicle_license_plate_classes_txt_path, help='Path of vehicle license plate classes txt.' +
                                                                                                                                                   '\n\tDefault: ' + str(default_vehicle_license_plate_classes_txt_path))
    parser.add_argument('-ltcp', '--license_plate_text_classes_txt_path', type=str, default=default_license_plate_text_classes_txt_path, help='Path of license plate text classes txt.' +
                                                                                                                                              '\n\tDefault: ' + str(default_license_plate_text_classes_txt_path))
    parser.add_argument('-vconf', '--vehicle_confidence_threshold', type=float, default=default_vehicle_confidence_threshold, help='Vehicle confidence threshold.' +
                                                                                                                                   '\n\tDefault: ' + str(default_vehicle_confidence_threshold))
    parser.add_argument('-viou', '--vehicle_iou_threshold', type=float, default=default_vehicle_iou_threshold, help='Vehicle iou threshold.' +
                                                                                                                    '\n\tDefault: ' + str(default_vehicle_iou_threshold))
    parser.add_argument('-lconf', '--license_plate_confidence_threshold', type=float, default=default_license_plate_confidence_threshold, help='License plate confidence threshold.' +
                                                                                                                                               '\n\tDefault: ' + str(default_license_plate_confidence_threshold))
    parser.add_argument('-liou', '--license_plate_iou_threshold', type=float, default=default_license_plate_iou_threshold, help='License plate iou threshold.' +
                                                                                                                                '\n\tDefault: ' + str(default_license_plate_iou_threshold))
    parser.add_argument('-ltconf', '--license_plate_text_confidence_threshold', type=float, default=default_license_plate_text_confidence_threshold, help='License plate text confidence threshold.' +
                                                                                                                                                          '\n\tDefault: ' + str(default_license_plate_text_confidence_threshold))
    parser.add_argument('-ltiou', '--license_plate_text_iou_threshold', type=float, default=default_license_plate_text_iou_threshold, help='License plate text iou threshold.' +
                                                                                                                                           '\n\tDefault: ' + str(default_license_plate_text_iou_threshold))
    parser.add_argument('--source', type=str, default=default_detect_source_path, help='Path of detect source. Support image, video or directory.' +
                                                                                       '\n* Image support format: ' + str(' '.join(support_image_format)) +
                                                                                       '\n* Video support format: ' + str(' '.join(support_video_format)) +
                                                                                       '\n* Webcam: number of camera' +
                                                                                       '\n\tDefault: ' + default_detect_source_path)
    parser.add_argument('-op', '--output_folder_path', type=str, default=default_output_folder_path, help='Path of output folder.' +
                                                                                                          '\n\tDefault: ' + default_output_folder_path)
    parser.add_argument('-show', '--show', action='store_false' if default_show else 'store_true', help='Show result.')
    parser.add_argument('-fps', '--fps', action='store_false' if default_show else 'store_true', help='Show fps.')
    parser.add_argument('-debug', '--debug', action='store_false' if default_show else 'store_true', help='Enable debug message.')
    return parser.parse_args()

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    #client.connect(broker, port)
    client.connect(serverIp, port)
    return client
    
def publish(client,msg):
    result = client.publish(topic, msg, qos=2)
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")
            
'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    from utils.python_package_utils import show_recommend_version

    show_recommend_version(recommend_package_dict=recommend_package_dict)
    np.set_printoptions(linewidth=2000)

    args = parse()
    client = connect_mqtt()
    client.loop_start()
    # create license plate recognition onnx
    license_plate_recognition = LicensePlateRecognition(vehicle_license_plate_detection_model_path=args.vehicle_license_plate_model_path,
                                                        license_plate_text_detection_model_path=args.license_plate_text_model_path,
                                                        warm_up_image_path=args.warm_up_image_path,
                                                        vehicle_license_plate_classes_txt_path=args.vehicle_license_plate_classes_txt_path,
                                                        vehicle_confidence_threshold=args.vehicle_confidence_threshold,
                                                        vehicle_iou_threshold=args.vehicle_iou_threshold,
                                                        license_plate_confidence_threshold=args.license_plate_confidence_threshold,
                                                        license_plate_iou_threshold=args.license_plate_iou_threshold,
                                                        license_plate_text_classes_txt_path=args.license_plate_text_classes_txt_path,
                                                        license_plate_text_confidence_threshold=args.license_plate_text_confidence_threshold,
                                                        license_plate_text_iou_threshold=args.license_plate_text_iou_threshold,
                                                        debug=args.debug,
                                                        show_init=True)
    license_plate_recognition.recognition_source(source=args.source,
                                                 output_folder_path=args.output_folder_path,
                                                 show=args.show,
                                                 show_fps=args.fps)
