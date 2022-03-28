from .reversible_edit import ReversibleEdit
from PIL import Image
import math

# Solution from user coproc: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr

class ReversibleRotation(ReversibleEdit):
    def __init__(self, rotation=0):
        self.rotation = rotation

    def forward(self, image):
        self.original_size = image.size
        img = image.rotate(self.rotation, expand=True, resample=Image.BILINEAR)
        #w_bb, h_bb = img.size
        #w_r, h_r = rotatedRectWithMaxArea(w,h,math.radians(self.rotation))
        #i_h = (w_bb-w_r)/2
        #i_v = (h_bb-h_r)/2
        #img = img.crop((i_h, i_v, w_bb-i_h, h_bb-i_v))
        return img

    def backward(self, image): 
        image = Image.fromarray(image)
        img = image.rotate(-self.rotation, expand=False)
        i_h = (img.size[0] - self.original_size[0]) / 2
        i_v = (img.size[1] - self.original_size[1]) / 2
        #img = img.crop((i_h, i_v, self.original_size[0]-i_h, self.original_size[1]-i_v))
        img = img.crop((i_h, i_v, img.size[0]-i_h, img.size[1]-i_v))
        return img