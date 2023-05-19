import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from typing import Union
from config import *
from utils import get_device

def get_mtcnn(
        thresholds: Union[int, int, int] = [0.6, 0.7, 0.7],
        margin: int = 0,
        min_face_size: int = 20,
        factor: float = 0.709,
    ) -> torch.nn.Module:
    """
    Returns the MTCNN model

    Parameters
    ----------
    thresholds : Union[int, int, int], optional
        Thresholds for the three steps of the detection pipeline, by default [0.6, 0.7, 0.7]

    margin : int, optional
        Margin around the detected face for the cropped output image, by default 0

    min_face_size : int, optional
        Minimum size of the face for the detection pipeline, by default 20

    factor : float, optional
        Factor for scaling the image pyramid, by default 0.709

    Returns
    -------
    torch.nn.Module
    """

    device = get_device()
    thresholds[-1] = DETECTION_THRESHOLD

    mtcnn = MTCNN(
        image_size=IMAGE_SIZE, margin=margin, min_face_size=min_face_size,
        thresholds=thresholds, factor=factor, post_process=True,
        device=device
    )

    return mtcnn


def get_facenet() -> torch.nn.Module:
    device = get_device()
    return InceptionResnetV1(pretrained='vggface2', device=device).eval()