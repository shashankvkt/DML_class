from .cub import CUBirds
from .import utils
from .base import BaseDataset


_type = {
    'cub': CUBirds
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
