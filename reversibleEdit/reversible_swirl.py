from .reversible_edit import ReversibleEdit
from PIL import Image
import numpy as np
from skimage.transform import swirl

# https://subscription.packtpub.com/book/application-development/9781785283932/1/ch01lvl1sec16/image-warping

class ReversibleSwirl(ReversibleEdit):
    def __init__(self):
        self.strength = 0.0
        self.center = None
        self.radius = 900

    def forward(self, image):
        pixels = np.asarray(image)
        out_img = (swirl(pixels, rotation=0, strength=self.strength, center=self.center,radius=self.radius)*255).astype(pixels.dtype)

        return Image.fromarray(out_img)

    def backward(self, image):
        pixels = np.asarray(image)
        out_img = (swirl(pixels, rotation=0, strength=-self.strength,  center=self.center,radius=self.radius)*255).astype(pixels.dtype)
        
        return Image.fromarray(out_img)