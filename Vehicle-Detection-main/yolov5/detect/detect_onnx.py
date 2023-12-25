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
"""
YOLO v5 Detect ONNX
"""

import os
import traceback
import onnxruntime
import cv2
import numpy as np
import torch
import time
from colorama import init, Fore

from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# initialize colorama
init(autoreset=True)


##########  Python Library Version ##########
recommend_package_dict = {'torch': '==1.8.0',
                          'onnxruntime': '==1.14.1',
                          'onnxruntime-gpu': '==1.14.1',
                          'opencv-python': '>=4.5.2.52',
                          'colorama': '>=0.4.5'}
#############################################


"""
Detect ONNX
"""
class DetectONNX:
    # ONNX model
    _model_path = None
    _model = None
    _input_name = None
    _input_size = 640
    _output_names = None
    _is_initialize = False

    # Torch
    _device = None

    # Confidence
    _confidence_threshold = 0.5

    # IOU
    _iou_threshold = 0.45

    # Result
    result_box_key = 'box'
    result_confidence_key = 'confidence'
    result_label_key = 'label'

    # Warm up
    _warm_up_image_path = 'images/warm_up_image.jpg'

    # Debug
    _debug = False


    def __init__(self, model_path, warm_up_image_path, confidence_threshold, iou_threshold, debug=True, show_init=True):
        '''
        Constructor

        Args:
            model_path: Path of model.
            debug: Enable debug message.
            show_init: Show initialize message.
        '''
        self._model_path = model_path
        self._warm_up_image_path = warm_up_image_path
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._debug = debug
        if show_init:
            print(Fore.GREEN + '\n========== YOLO V5 Detect ONNX ==========')
            print(Fore.GREEN + 'model path: ', self._model_path)
            print(Fore.GREEN + 'warm up image path: ', self._warm_up_image_path)
            print(Fore.GREEN + 'confidence threshold: ', self._confidence_threshold)
            print(Fore.GREEN + 'iou threshold: ', self._iou_threshold)
            print(Fore.GREEN + 'debug: ', self._debug)

        self._init_device()
        self._load_onnx_model()
        self._warn_up()

        if show_init:
            print(Fore.GREEN + '=========================================\n')


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
        predict_time = 0
        success = False

        try:
            start_time = time.time()
            if self._debug:
                print('\n********** Detect **********')

            if self._model is None:
                raise Exception('ONNX model is not initialize.')

            if image is None:
                if image_path is None:
                    raise Exception('Image and image path is None.')
                if not os.path.exists(image_path):
                    raise Exception('Image (' + str(image_path) + ') is not exist.')

                # load image
                image = cv2.imread(image_path)

            # predict
            results, predict_time = self._predict(image)
            success = True
            detect_time = time.time() - start_time

        except Exception as e:
            print(Fore.RED + 'Detect error. ' + str(e) +
                  '\n' + str(traceback.format_exc()))
            results = []

        finally:
            if self._debug:
                if success:
                    print('Detect object number: ' + str(len(results)))
                    for result in results:
                        print(result)
                    print('Predict time: {:.4f}s.  Detect time: {:.4f}s.'.format(predict_time, detect_time))
                print('****************************\n')
            return results, predict_time, detect_time


    def _init_device(self):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def _load_onnx_model(self):
        try:
            print('\nStarting load onnx model (' + self._model_path + ')...')
            self._is_initialize = False
            start_time = time.time()

            if not os.path.exists(self._model_path):
                raise Exception('ONNX model (' + str(self._model_path) + ' is not exist.')

            self._model = onnxruntime.InferenceSession(self._model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self._input_name = self._model.get_inputs()[0].name
            self._input_type = self._model.get_inputs()[0].type
            self._input_size = self._model.get_inputs()[0].shape[2]
            self._output_names = [x.name for x in self._model.get_outputs()]

        except Exception as e:
            print(Fore.RED + 'Load ONNX model failed. ' + str(e) +
                  '\n' + str(traceback.format_exc()))
            self._model = None
            self._is_initialize = False
            return

        print('Input: ', self._model.get_inputs()[0])
        print('Output: ', self._model.get_outputs()[0])
        print(Fore.GREEN + 'Load ONNX model success. ', 'Cost time: ' + str(time.time() - start_time) + 's.')
        self._is_initialize = True


    def _warn_up(self):
        try:
            print('\nStarting warm up...')
            start_time = time.time()

            if not os.path.exists(self._warm_up_image_path):
                raise Exception('Image (' + str(self._warm_up_image_path) + ') is not exist.')

            image = cv2.imread(self._warm_up_image_path)
            _ = self._predict(image)

            print(Fore.GREEN + 'Warm up success. ', 'Cost time: ' + str(time.time() - start_time) + 's.')

        except Exception as e:
            print(Fore.RED + 'Warm up failed. ' + str(e) +
                  '\n' + str(traceback.format_exc()))


    def _predict(self, image):
        results = []
        original_image = image.copy()

        # transform image
        image = letterbox(image, self._input_size, stride=32, auto=False)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)
        if self._input_type == 'tensor(float16)':
            image = np.array([image], dtype=np.float16)
        else:
            image = np.array([image], dtype=np.float32)
        image /= 255

        # predict
        start_predict_time = time.time()
        output = self._model.run(self._output_names, {self._input_name: image})
        predict_time = time.time() - start_predict_time

        output = self._from_numpy(np.float32(output[0])) if len(output) == 1 else [self._from_numpy(x) for x in output]
        output = non_max_suppression(output, conf_thres=self._confidence_threshold, iou_thres=self._iou_threshold, classes=None, agnostic=False, max_det=1000)
        output = output[0].cpu().numpy()

        boxes = output[:, :4]
        confidences = output[:, 4:5]
        labels = output[:, 5:6]

        # scale to real size
        boxes = scale_boxes(self._model.get_inputs()[0].shape[2:4],
                            boxes,
                            original_image.shape)

        for label, box, confidence in zip(labels, boxes, confidences):
            results.append({self.result_label_key: label[0],
                            self.result_box_key: box,
                            self.result_confidence_key: confidence[0]})
        return results, predict_time


    def _from_numpy(self, x):
        return torch.from_numpy(x).to(self._device) if isinstance(x, np.ndarray) else x




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

default_detect_source_path = '../../images/test_license_plate_recognition.png'
default_output_folder_path = '../../outputs'
default_show = False

support_image_format = ['.jpg', '.jpeg', '.png', '.bmp', '.dib', '.webp', '.sr', '.ras', '.tiff', '.tif']
support_video_format = ['.mp4', '.avi']



'''
=============================
Args Parse
=============================
'''
def parse():
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(description='YOLO v5 Detect ONNX', formatter_class=RawTextHelpFormatter)
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
                                                                                       '\n\tDefault: ' + default_detect_source_path)
    parser.add_argument('-op', '--output_folder_path', type=str, default=default_output_folder_path, help='Path of output folder.' +
                                                                                                          '\n\tDefault: ' + default_output_folder_path)
    parser.add_argument('-show', '--show', action='store_false' if default_show else 'store_true', help='Show result.')
    return parser.parse_args()


def detect_image(image_path=None, bgr_image=None, output_folder_path=None, show=True):
    if bgr_image is None:
        image = cv2.imread(image_path)
    else:
        image = bgr_image.copy()

    # predict
    results, _, _ = detect_onnx.detect(image=image)

    # draw box and label class
    for result in results:
        label = result[DetectONNX.result_label_key].astype(int)
        box = result[DetectONNX.result_box_key].astype(int)

        # draw label class
        text = classes[label]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(image, (box[0], box[1]), (box[0] + text_size[0], box[1] - text_size[1]), colors[label], -1)
        cv2.putText(image, classes[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # draw box
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[label], 2)

    # save result image
    if output_folder_path is not None:
        save_path = os.path.join(output_folder_path, os.path.splitext(os.path.basename(image_path))[0] + '.jpg')
        cv2.imwrite(save_path, image)

    # show result image
    if show:
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def detect_video(video_path, output_folder_path=None, show=True):
    video = cv2.VideoCapture(video_path)

    result_video = None
    if output_folder_path is not None:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        save_path = os.path.join(output_folder_path, os.path.splitext(os.path.basename(video_path))[0] + '.mp4')
        result_video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print(Fore.RED + 'Can not receive frame (end frame). Exiting...')
            break

        # detect
        frame = detect_image(bgr_image=frame, show=False)

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
Main
=============================
'''
if __name__ == '__main__':
    from utils.python_package_utils import show_recommend_version
    show_recommend_version(recommend_package_dict=recommend_package_dict)
    np.set_printoptions(linewidth=2000, edgeitems=10)

    args = parse()

    # create detect onnx
    detect_onnx = DetectONNX(model_path=args.model_path,
                             warm_up_image_path=args.warm_up_image_path,
                             confidence_threshold=args.confidence_threshold,
                             iou_threshold=args.iou_threshold)

    # get classes
    classes = []
    with open(args.classes_txt_path, 'r') as file:
        for line in file.readlines():
            classes.append(line.replace('\n', ''))
    print('Label classes: ', classes)

    # generate classes color
    import random
    colors = set()
    for _ in range(len(classes)):
        colors.add((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    colors = list(colors)


    # create save result image folder
    if args.output_folder_path is not None:
        # create output folder
        if not os.path.exists(args.output_folder_path):
            os.mkdir(args.output_folder_path)


    # get detect file list
    detect_file_list = []
    if os.path.isfile(args.source):
        detect_file_list.append(args.source)

    else:
        for file_name in os.listdir(args.source):
            detect_file_list.append(os.path.join(args.source, file_name))


    for detect_file in detect_file_list:
        _, file_extension = os.path.splitext(detect_file)

        if file_extension in support_image_format:
            ''' Image '''
            # image = cv2.imread(detect_file)
            _ = detect_image(image_path=detect_file, output_folder_path=args.output_folder_path, show=args.show)

        elif file_extension in support_video_format:
            ''' Video '''
            detect_video(video_path=detect_file, output_folder_path=args.output_folder_path, show=args.show)