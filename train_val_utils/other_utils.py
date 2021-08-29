import torch
import random
import numpy as np
from torch.backends import cudnn
import os
import glob
import time


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reduce randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True

def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file):
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()