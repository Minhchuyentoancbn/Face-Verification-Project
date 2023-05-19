IMAGE_SIZE = 160  # All images will be resized to this size
DETECTION_THRESHOLD = 0.9999  # Probability threshold to detect a face
VERIFICATION_THRESHOLD = 0.83  # Distance threshold to decide whether faces belong to the same person

SEED = 42

DATA_PATH = 'data/'
CASIA_PATH = 'data/CASIA-WebFace/'
LFW_PATH = 'data/lfw/lfw_cropped'
LFW_PAIRS_PATH = 'data/lfw/pairs.txt'
LOG_DIR = 'runs/'


# casia_cropped_path = os.path.join(DATA_PATH, 'CASIA-WebFace-cropped/')
# casia_cropped_path = '/kaggle/input/casia-webface-cropped-with-mtcnn/CASIA-WebFace-cropped'
casia_cropped_path = '/kaggle/input/casia-webface-mtcnn-v2/CASIA-WebFace-cropped'

MIN_MOMENTUM = 0.8

def lr_update_rule(step):
    if step < 15:
        return 1#(step + 1) / 10
    else:
        return 0.1