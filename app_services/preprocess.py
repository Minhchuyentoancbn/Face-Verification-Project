import cv2
import numpy as np

from PIL import Image
from utils import empty_folder
from typing import List
from models import get_mtcnn
from config import *

import torch
import torch.nn as nn

mtcnn = get_mtcnn()

def process_video(
    path: str,
    save: bool = False,
    prob_threshold: float = DETECTION_THRESHOLD
) -> List[torch.Tensor]:
    """
    Processes the video and saves the frames in the app_data/temp/picture folder
    Crop faces from the frames and save them in the app_data/temp/cropped folder
    Get maximum 10 images from the video

    Parameters
    ----------
    path : str
        Path to the video file

    save : bool, optional
        Whether to save the frames or not, by default False

    prob_threshold : float, optional
        Probability threshold for face detection

    Returns
    -------
    List[torch.Tensor]
        List of images. Each image is a tensor of shape (160, 160, 3)

    """
    images_list = []

    # Empty the app_data/temp/picture and app_data/temp/cropped folders
    empty_folder("app_data/temp/picture")
    empty_folder("app_data/temp/cropped")

    # Read the video file
    try:
        cap = cv2.VideoCapture(path)
    except:
        print("Error reading video file")
        print("Please check the path and try again")
        return []
    
    # Get the frame and number of frames
    # fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # set the sample time between samples
    sample_time = 1000 / 3  # 1000 ms = 1 second

    # set the initial position
    pos_msec = 0


    # iterate through all frames and convert each to PIL image
    for i in range(frame_count):
        # set the position to the next sample time
        cap.set(cv2.CAP_PROP_POS_MSEC, pos_msec)
        # read the frame
        ret, frame = cap.read()

        if ret:
            # convert the frame from BGR to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # convert the frame to PIL image
            image = Image.fromarray(frame_rgb)

            # save the PIL image to file or process it further
            if save:
                image.save(f'app_data/temp/picture/frame_{i}.jpg')

            # update the position for the next sample
            pos_msec += sample_time

            # Crop the faces from the frames
            x_aligned, prob = mtcnn(image, return_prob=True)
            if prob is not None:
                if prob > prob_threshold:
                    # Add the image to the list
                    images_list.append(x_aligned)
                    if save:
                        # Convert from tensor to PIL image and save
                        x_aligned = x_aligned.permute(1, 2, 0).to('cpu').numpy()
                        x_aligned = x_aligned * 128 + 127.5
                        x_aligned = x_aligned.astype(np.uint8)
                        x_aligned = Image.fromarray(x_aligned)

                        # Save the cropped image
                        x_aligned.save(f'app_data/temp/cropped/frame_{i}.jpg')
        else:
            break

    # release the video file
    cap.release()

    return images_list