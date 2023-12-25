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
detect.py
'''
import sys
import os
import cv2
import random
import traceback
import numpy as np
from tqdm import tqdm
from colorama import init, Fore

sys.path.append('../../')

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


class Detect:
    # Model
    _model_path = None
    _model = None
    _is_initialize = False

    # Classes list
    _classes = []
    _classes_color = set()

    # Torch
    _device = None

    # Confidence
    _confidence_threshold = 0.5

    # IOU
    _iou_threshold = 0.4

    # Warm up
    _warm_up_image_path = '../../images/warm_up_image.jpg'

    # Result
    result_box_key = 'box'
    result_confidence_key = 'confidence'
    result_label_key = 'label'

    # Output
    _output_folder_path = None

    # Debug
    _debug = False


    def __init__(self, model_path, warm_up_image_path, confidence_threshold, iou_threshold, debug=_debug, show_init=True):
        '''
        Constructor

        Args:
            model_path: Path of model
            warm_up_image_path: Path of warm up image
            confidence_threshold: Confidence threshold
            iou_threshold: IOU threshold
            debug: Enable debug message
            show_init: Show initialize message
        '''
        self._model_path = model_path
        self._warm_up_image_path = warm_up_image_path
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._debug = debug
        if show_init:
            print(Fore.GREEN + '\n==========     YOLO V5 Detect    ==========')
            print(Fore.GREEN + 'model path: ', self._model_path)
            print(Fore.GREEN + 'warm up image path: ', self._warm_up_image_path)
            print(Fore.GREEN + 'confidence threshold: ', self._confidence_threshold)
            print(Fore.GREEN + 'iou threshold: ', self._iou_threshold)
            print(Fore.GREEN + 'debug: ', self._debug)

        self._init_model()
        self._init_output_folder()

        if show_init:
            print(Fore.GREEN + '===========================================\n')


    def is_initialize(self):
        return self._is_initialize


    def detect(self, image=None, image_path=None):
        '''
        Detect

        Args:
            image (numpy): The BGR image you want to use for detection.
            image_path (str): The image path you want to use for detection.

        Returns: list containing all the bounding boxes detected with their
                    label index (int):
                    box (list[float]): [top x, top y, button x, bottom y]
                    confidence (float): 0 ~ 1
        '''
        if self._model is None:
            print(Fore.RED + 'Model is not initialize, can not detect!!')
            return None

        return self._model.detect(image=image, image_path=image_path)


    def detect_source(self, source, classes_txt_path, output_folder_path=None, show=True, show_fps=False):
        print(Fore.YELLOW + '\n**********   Detect Source   **********')
        print(Fore.YELLOW + 'source: ', source)
        print(Fore.YELLOW + 'classes txt path: ', classes_txt_path)
        print(Fore.YELLOW + 'output folder path: ', output_folder_path)
        print(Fore.YELLOW + 'show: ', str(show))
        print(Fore.YELLOW + 'show fps: ', str(show_fps))

        try:
            if self._model is None:
                raise Exception('Model is not initialize, can not detect!!')

            self._init_classes(classes_txt_path=classes_txt_path)
            self._init_classes_color()

            # get source list
            detect_file_list = []
            is_camera = False
            if os.path.isfile(source):
                detect_file_list.append(source)
            elif os.path.isdir(source):
                for file_name in os.listdir(source):
                    detect_file_list.append(os.path.join(source, file_name))
            else:
                is_camera = True
            print(Fore.YELLOW + 'Source: ', 'Camera ' + str(source) if is_camera else detect_file_list)

            if is_camera:
                self._detect_video(video_path=int(args.source), is_camera=is_camera, output_folder_path=output_folder_path, show=show, show_fps=show_fps)
            else:
                for detect_file in detect_file_list:
                    _, file_extension = os.path.splitext(detect_file)

                    if file_extension in support_image_format:
                        ''' Image '''
                        _ = self._detect_image(image_path=detect_file, output_folder_path=output_folder_path, show=show, show_fps=show_fps)

                    elif file_extension in support_video_format:
                        ''' Video '''
                        self._detect_video(video_path=detect_file, output_folder_path=output_folder_path, show=show, show_fps=show_fps)

        except Exception as ex:
            print(Fore.RED + 'Detect source error. ' + str(ex) + '\n' + str(traceback.format_exc()))

        finally:
            print(Fore.YELLOW + '***************************************\n')


    def _init_model(self):
        _, file_extension = os.path.splitext(self._model_path)
        if file_extension == '.onnx':
            from yolov5.detect.detect_onnx import DetectONNX
            self._model = DetectONNX(model_path=self._model_path,
                                     warm_up_image_path=self._warm_up_image_path,
                                     confidence_threshold=self._confidence_threshold,
                                     iou_threshold=self._iou_threshold,
                                     debug=self._debug,
                                     show_init=False)

        elif file_extension == '.tflite':
            from yolov5.detect.detect_tflite import DetectTFLite
            self._model = DetectTFLite(model_path=self._model_path,
                                       warm_up_image_path=self._warm_up_image_path,
                                       confidence_threshold=self._confidence_threshold,
                                       iou_threshold=self._iou_threshold,
                                       debug=self._debug,
                                       show_init=False)

        elif file_extension == '.engine':
            from yolov5.detect.detect_trt_engine import DetectTRTEngine
            self._model = DetectTRTEngine(model_path=self._model_path,
                                          warm_up_image_path=self._warm_up_image_path,
                                          confidence_threshold=self._confidence_threshold,
                                          iou_threshold=self._iou_threshold,
                                          debug=self._debug,
                                          show_init=False)

        else:
            print(Fore.RED + file_extension + ' model is not support!!')
            self._is_initialize = False

        if self._model is not None:
            self._is_initialize = self._model.is_initialize()


    def _init_classes(self, classes_txt_path):
        if classes_txt_path is not None:
            with open(classes_txt_path, 'r') as file:
                for line in file.readlines():
                    self._classes.append(line.replace('\n', ''))
            print('\nLabel classes: ', self._classes)


    def _init_classes_color(self):
        for _ in range(len(self._classes)):
            self._classes_color.add((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        self._classes_color = list(self._classes_color)


    def _init_output_folder(self):
        if self._output_folder_path is not None:
            # create output folder
            if not os.path.exists(self._output_folder_path):
                os.mkdir(self._output_folder_path)


    def _detect_image(self, image_path=None, bgr_image=None, output_folder_path=None, show=True, show_fps=False):
        if bgr_image is None:
            image = cv2.imread(image_path)
        else:
            image = bgr_image.copy()

        # predict
        results, _, detect_time = self.detect(image=image)
        fps = 1 / detect_time

        # draw box and label class
        for result in results:
            label = result[self._model.result_label_key].astype(int)
            box = result[self._model.result_box_key].astype(int)

            # draw label class
            text = self._classes[label]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(image, (box[0], box[1]), (box[0] + text_size[0], box[1] - text_size[1]), self._classes_color[label], -1)
            cv2.putText(image, self._classes[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # draw box
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), self._classes_color[label], 2)

        # FPS
        if show_fps:
            cv2.putText(image, "FPS: " + format(fps, '.1f'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # save result image
        if output_folder_path is not None:
            save_path = os.path.join(output_folder_path, os.path.splitext(os.path.basename(image_path))[0] + '.jpg')
            cv2.imwrite(save_path, image)

        # show result image
        if show:
            cv2.imshow('Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image, fps


    def _detect_video(self, video_path, is_camera=False, output_folder_path=None, show=True, show_fps=False):
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
            frame, fps = self._detect_image(bgr_image=frame, show=False, show_fps=show_fps)
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
        progress.close()

        video.release()
        cv2.destroyAllWindows()
        if result_video is not None:
            result_video.release()




'''
=============================
Parse Default
=============================
'''
default_onnx_model_path = '../../models_zoo/yolov5/n/vehicle-license-plate-detection-n-alan-640-v2/vehicle-license-plate-detection-n-alan-640-v2.onnx'
default_warm_up_image_path = '../../images/warm_up_image.png'
default_classes_txt_path = '../../classes/classes.txt'
default_confidence_threshold = 0.5
default_iou_threshold = 0.4

# source
default_detect_source_path = '../../images/test_license_plate_recognition.png'
support_image_format = ['.jpg', '.jpeg', '.png', '.bmp', '.dib', '.webp', '.sr', '.ras', '.tiff', '.tif']
support_video_format = ['.mp4', '.avi']

# output
default_output_folder_path = '../../outputs'
default_show = False
default_show_fps = False

# debug
default_debug = False


'''
=============================
Args Parse
=============================
'''
def parse():
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(description='YOLO v5 Detect', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-mp', '--model_path', type=str, default=default_onnx_model_path, help='Path of model.' +
                                                                                               '\n\tDefault: ' + default_onnx_model_path)
    parser.add_argument('-wip', '--warm_up_image_path', type=str, default=default_warm_up_image_path, help='Path of warm up image.' +
                                                                                                           '\n\tDefault: ' + default_warm_up_image_path)
    parser.add_argument('-cp', '--classes_txt_path', type=str, default=default_classes_txt_path, help='Path of classes txt.' +
                                                                                                      '\n\tDefault: ' + default_classes_txt_path)
    parser.add_argument('-conf', '--confidence_threshold', type=float, default=default_confidence_threshold, help='Confidence threshold.' +
                                                                                                                  '\n\tDefault: ' + str(default_confidence_threshold))
    parser.add_argument('-iou', '--iou_threshold', type=float, default=default_iou_threshold, help='IOU threshold.' +
                                                                                                   '\n\tDefault: ' + str(default_iou_threshold))
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


'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    from utils.python_package_utils import show_recommend_version
    show_recommend_version(recommend_package_dict=recommend_package_dict)
    np.set_printoptions(linewidth=2000, edgeitems=10)

    args = parse()

    # create detect
    detect = Detect(model_path=args.model_path,
                    warm_up_image_path=args.warm_up_image_path,
                    confidence_threshold=args.confidence_threshold,
                    iou_threshold=args.iou_threshold,
                    debug=args.debug,
                    show_init=True)
    detect.detect_source(source=args.source,
                         classes_txt_path=args.classes_txt_path,
                         output_folder_path=args.output_folder_path,
                         show=args.show,
                         show_fps=args.fps)
