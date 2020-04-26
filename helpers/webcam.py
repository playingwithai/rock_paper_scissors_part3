from contextlib import contextmanager

import cv2


@contextmanager
def opencv_video_capture(webcam_index):
    camera = cv2.VideoCapture(webcam_index)
    yield camera
    camera.release()


def get_webcam_frame(camera, width=300, height=300):
    _, frame = camera.read()
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame
