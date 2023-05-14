import numpy as np
import pandas as pd
import os

from PIL import Image
from preprocess import process_video
from utils import get_device

import torch

def register(
    video_path: str,
    userid: str,
    prob_threshold: float = 0.99,
    mtcnn: torch.nn.Module = None,
    facenet: torch.nn.Module = None,
    num_images: int = 10
):
    """
    Save the embedding tensor and threshold value for userid

    Parameters
    ----------
    video_path : str
        Path to the video file

    userid : str
        User ID

    prob_threshold : float, optional
        Probability threshold for face detection, by default 0.99

    mtcnn : torch.nn.Module, optional
        MTCNN model, by default None

    facenet : torch.nn.Module, optional
        Facenet model, by default None

    num_images : int, optional
        Number of images to save, by default 10
    """
    # Check the parameters
    if facenet is None:
        print("Please provide Facenet model")
        return

    images = process_video(
        video_path,  
        crop=True,
        prob_threshold=prob_threshold,
        mtcnn=mtcnn
    )

    if len(images) == 0:
        print("No faces found in the video")
        return

    # Get the device
    device = get_device()

    # Check if the app_data/database/userid folder exists
    if not os.path.exists(f"app_data/database/{userid}"):
        os.makedirs(f"app_data/database/{userid}")

    # compute the mean of the images in the cropped folder
    images = torch.stack(images).float().to(device)
    facenet = facenet.eval().to(device)
    embeddings = facenet(images).detach().cpu()


    # select top 10 images that closely resemble the mean image
    mean_embedding = embeddings.mean(dim=0)
    indices = []
    distances = torch.norm(embeddings - mean_embedding, dim=1)
    # Get minimum indices
    indices.append(torch.argmin(distances).item())

    for i in range(num_images - 1):
        estimated_mean_embedding = (embeddings[indices].sum(dim=0) + embeddings) / (i + 2)
        distances = torch.norm(estimated_mean_embedding - mean_embedding, dim=1)
        indices.append(torch.argmin(distances).item())


    # save the images to app_data/database/userid folder
    for i, index in enumerate(indices):
        # Get the image in app_data/temp/cropped folder
        filename = os.listdir("app_data/temp/cropped")[index]
        img = Image.open(f"app_data/temp/cropped/{filename}")
        img.save(f"app_data/database/{userid}/{i}.jpg")

    # Free the memory
    del images
    del embeddings
    torch.cuda.empty_cache()

    return


def verification(
    userid: str,
    video_path: str,
    mtcnn: torch.nn.Module,
    facenet: torch.nn.Module,
    prob_threshold: float = 0.99,
    threshold: float = 1.256
) -> bool:
    """
    Verifies the user's identity

    Parameters
    ----------
    userid : str
        User ID

    video_path : str
        Path to the video file

    mtcnn : torch.nn.Module
        MTCNN model

    facenet : torch.nn.Module
        Facenet model

    prob_threshold : float, optional
        Probability threshold for face detection, by default 0.99

    threshold : float, optional
        Threshold value for L2-distance, by default 1.256

    Returns
    -------
    bool
        Whether the user is verified or not
    """
    # Set the device
    device = get_device()
    facenet = facenet.eval().to(device)

    # Check if the app_data/database/userid folder exists
    if not os.path.exists(f"app_data/database/{userid}"):
        print("User not registered")
        return False

    # Get the images from the video
    images = process_video(
        video_path,
        crop=True,
        prob_threshold=prob_threshold,
        mtcnn=mtcnn
    )

    if len(images) == 0:
        print("No faces found in the video")
        return False
    
    # Read user's images
    user_images = []
    for filename in os.listdir(f"app_data/database/{userid}"):
        img = Image.open(f"app_data/database/{userid}/{filename}")
        img = torch.tensor((np.array(img) -127.5) * 0.0078125).permute(2, 0, 1)
        user_images.append(img)
    
    # Get mean embedding of user's images
    user_images = torch.stack(user_images).float().to(device)
    user_embeddings = facenet(user_images).mean(dim=0).detach().cpu()

    # Get the embeddings
    images = torch.stack(images).float().to(device)
    embeddings = facenet(images).detach().cpu()

    # Compute the mean L2-distance
    distances = torch.norm(embeddings - user_embeddings, dim=1)
    mean_distance = distances.mean()

    print(f"Mean distance: {mean_distance}")

    # Free the memory
    del images
    del embeddings
    del user_images
    del user_embeddings
    torch.cuda.empty_cache()

    # Check if the user is verified
    if mean_distance < threshold:
        return True
    else:
        return False