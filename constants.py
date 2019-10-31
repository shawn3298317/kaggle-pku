import numpy as np

PATH = "/projectnb/cs542/basement-scientists/pku-auto-drive/"

CAMERA_MATRIX = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

IMG_WIDTH = 2048 - 512
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8
IMG_SHAPE = (2710, 3384, 3)
