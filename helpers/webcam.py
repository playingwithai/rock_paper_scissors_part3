from contextlib import contextmanager

import cv2
import pygame


@contextmanager
def opencv_video_capture(webcam_index):
    camera = cv2.VideoCapture(webcam_index)
    yield camera
    camera.release()


def opencv_to_pygame_image(image):
    image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(image.tostring(), image.shape[1::-1], "RGB")
