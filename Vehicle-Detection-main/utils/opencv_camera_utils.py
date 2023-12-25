#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   Copyright (C) 2022 MSI-FUNTORO
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
import cv2

class OpenCVCameraUtils:
    def open(self, camera_id, width=None, height=None, fps=None, codec=None):
        print('\n************ Open Camera ************')
        print('Camera ID: ', camera_id)
        print('Size: ', width, 'x', height)
        print('FPS: ', fps)
        print('Codec: ', codec)

        try:
            print('Opening...')
            camera = cv2.VideoCapture(camera_id)
            print('Open done.')
        except Exception as ex:
            print('Open error. ', ex)
            return None

        try:
            print('Set camera config...')
            if width is not None: camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None: camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps is not None: camera.set(cv2.CAP_PROP_FPS, fps)
            if codec is not None: camera.set(cv2.CAP_PROP_FOURCC, codec)
            print('Set camera config done.')
        except Exception as ex:
            print('Set camera config error. ', ex)
            return None

        self.show_camera_config(camera)
        print('*************************************\n')
        return camera


    def show_camera_config(self, camera:cv2.VideoCapture):
        if camera is None:
            return

        try:
            width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = camera.get(cv2.CAP_PROP_FPS)
            codec = int(camera.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

            print('Size: ', width, 'x', height)
            print('FPS: ', fps)
            print('Codec: ', codec)

        except Exception as ex:
            print('Show camera config error. ', ex)


    def show(self, camera:cv2.VideoCapture):
        if camera is not None:
            while camera.isOpened():
                ret, frame = camera.read()
                if not ret:
                    break

                cv2.imshow('Camera', frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

            camera.release()
            cv2.destroyAllWindows()


    def show_hls(self, camera:cv2.VideoCapture):
        from skimage.feature import local_binary_pattern
        if camera is not None:
            while camera.isOpened():
                ret, frame = camera.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                # print(frame[:, :, 1])
                # frame[:, :, 1] = 0
                # frame[:, :, 2] = 0
                # frame = cv2.cvtColor(frame, cv2.COLOR_HLS2BGR)
                # frame = frame[:, :, 1]
                # lbp = local_binary_pattern(image=frame, P=8, R=1)

                cv2.imshow('Camera Y', local_binary_pattern(frame[:, :, 2], 2, 1))
                # cv2.imshow('Camera Cr', local_binary_pattern(frame[:, :, 1], 8, 1))
                # cv2.imshow('Camera Cb', local_binary_pattern(frame[:, :, 2], 8, 1))

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

            camera.release()
            cv2.destroyAllWindows()



'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    import numpy as np
    np.set_printoptions(linewidth=2000)

    opencv_camera_utils = OpenCVCameraUtils()
    camera = opencv_camera_utils.open(camera_id=0, width=2560, height=720, codec=1196444237.0)
    # opencv_camera_utils.show(camera=camera)
    opencv_camera_utils.show_hls(camera=camera)
