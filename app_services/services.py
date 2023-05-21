import os

from .preprocess import process_video
from utils import get_device
from config import *
from models import get_facenet

import torch


facenet = get_facenet()
device = get_device()


def register(
    video_path: str,
    userid: str,
    prob_threshold: float = DETECTION_THRESHOLD
) -> bool:
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

    Returns
    -------
    bool
        Whether the user is registered or not
    """
    # Check the parameters
    images = process_video(
        video_path,  
        prob_threshold=prob_threshold
    )

    if len(images) == 0:
        print("No faces found in the video")
        return False

    # Check if the app_data/database/userid folder exists
    if not os.path.exists(f"app_data/database/{userid}"):
        os.makedirs(f"app_data/database/{userid}")

    # compute the mean of the images in the cropped folder
    images = torch.stack(images).float().to(device)
    embeddings, _ = facenet(images)
    embeddings = embeddings.detach().cpu()

    # Save the embeddings
    torch.save(embeddings, f"app_data/database/{userid}/embeddings.pt")

    # Free the memory
    del images, embeddings
    torch.cuda.empty_cache()

    print("User registered successfully")
    return True


def verification(
    userid: str,
    video_path: str,
    prob_threshold: float = DETECTION_THRESHOLD,
    threshold: float = VERIFICATION_THRESHOLD
):
    """
    Verifies the user's identity

    Parameters
    ----------
    userid : str
        User ID

    video_path : str
        Path to the video file

    prob_threshold : float, optional
        Probability threshold for face detection

    threshold : float, optional
        Threshold value for cosine similarity

    Returns
    -------
    bool
        Whether the user is verified or not

    similarity : float
        Cosine similarity between the user's embeddings and the embeddings of the images in the video
    """

    # Check if the app_data/database/userid folder exists
    if not os.path.exists(f"app_data/database/{userid}"):
        print("User not registered")
        return False

    # Get the images from the video
    images = process_video(
        video_path,
        prob_threshold=prob_threshold
    )

    if len(images) == 0:
        print("No faces found in the video")
        return False
    
    # Read user's embeddings
    user_embeddings = torch.load(f"app_data/database/{userid}/embeddings.pt")

    # Get the embeddings of the images
    images = torch.stack(images).float().to(device)
    images_embeddings, _ = facenet(images)
    images_embeddings = images_embeddings.detach().cpu()

    # Compute the cosine similarity between all pairs of images and user's embeddings
    # Matrix multiplication of images_embeddings and user_embeddings
    similarity = torch.mm(images_embeddings, user_embeddings.t()).mean().item()
    if similarity >= threshold:
        print("User verified successfully")
        torch.cuda.empty_cache()
        return True, similarity
    
    print("User not verified")
    torch.cuda.empty_cache()
    return False, similarity


def identification(
    video_path: str,
    prob_threshold: float = DETECTION_THRESHOLD,
    threshold: float = VERIFICATION_THRESHOLD
):
    """
    Identifies the person in the video

    Parameters
    ----------
    video_path : str
        Path to the video file

    prob_threshold : float, optional
        Probability threshold for face detection

    threshold : float, optional
        Threshold value for cosine similarity

    Returns
    -------
    str
        User ID of the person in the video
    """

    user_list = os.listdir("app_data/database")
    user_similarity = dict()

    # Get the images from the video
    images = process_video(
        video_path,
        prob_threshold=prob_threshold
    )

    if len(images) == 0:
        print("No faces found in the video")
        return None

    # Get the embeddings of the images
    images = torch.stack(images).float().to(device)
    images_embeddings, _ = facenet(images)
    images_embeddings = images_embeddings.detach().cpu()

    # Iterate over all the users
    for user in user_list:
        # Read user's embeddings
        user_embeddings = torch.load(f"app_data/database/{user}/embeddings.pt")

        # Compute the cosine similarity between all pairs of images and user's embeddings
        # Matrix multiplication of images_embeddings and user_embeddings
        similarity = torch.mm(images_embeddings, user_embeddings.t()).mean().item()

        if similarity >= threshold:
            user_similarity[user] = similarity

    if len(user_similarity) == 0:
        print("No user found")
        return ""
    
    # Get the user with the highest similarity
    user = max(user_similarity, key=user_similarity.get)
    print(f"User {user} identified successfully with similarity {user_similarity[user]}")
    torch.cuda.empty_cache()
    return user