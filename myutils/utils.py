import os
import glob
import re
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import math
import time

from myutils.vgg16 import Vgg16

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


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False, np_array=False, rotation=0, reversible_edit=None):
    img = Image.open(filename).convert('RGB')
    if rotation != 0:
        w, h = img.size
        img = img.rotate(rotation, expand=True, resample=Image.BILINEAR)
        w_bb, h_bb = img.size
        w_r, h_r = rotatedRectWithMaxArea(w,h,math.radians(rotation))
        i_h = (w_bb-w_r)/2
        i_v = (h_bb-h_r)/2
        img = img.crop((i_h, i_v, w_bb-i_h, h_bb-i_v))

    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)
    
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    if reversible_edit:
        img = reversible_edit(img)
    img = np.array(img).transpose(2, 0, 1)
    if not np_array:
        img = torch.from_numpy(img).float()
    return img


def tensor_load_rgbimage_hw(filename, width=None, height=None, np_array=False):
    img = Image.open(filename).convert('RGB')
    img = img.resize((width, height), Image.ANTIALIAS)
    
    img = np.array(img).transpose(2, 0, 1)
    if not np_array:
        img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y, active_pixelcount=-1):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    if active_pixelcount <= 0:
        gram = features.bmm(features_t) / (ch * h * w)
    else:
        # divide by a number of pixels which are actually used in the image
        gram = features.bmm(features_t) / (ch * active_pixelcount)
    return gram


def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    tensortype = eval(batch.type())
    mean = tensortype(batch.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch - mean


def add_imagenet_mean_batch(batch):
    """Add ImageNet mean pixel-wise from a BGR image."""
    tensortype = eval(batch.type())
    mean = tensortype(batch.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch + mean


def imagenet_clamp_batch(batch, low, high):
    batch[:, 0, :, :].data.clamp_(low - 103.939, high - 103.939)
    batch[:, 1, :, :].data.clamp_(low - 116.779, high - 116.779)
    batch[:, 2, :, :].data.clamp_(low - 123.680, high - 123.680)


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def init_vgg16(model_folder):
    """load the vgg16 model feature"""
    if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
        from torch.utils.serialization import load_lua #only available in torch < 1.0
        if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
            os.system('wget --no-check-certificate http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(model_folder, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))


def total_variation_loss(x, tv_weight=1e-4):
    assert x.dim() == 4
    img_width = x.size(3)
    img_height = x.size(2)
    a = torch.pow(x[:, :, :img_height - 1, :img_width - 1] - x[:, :, 1:, :img_width - 1], 2)
    b = torch.pow(x[:, :, :img_height - 1, :img_width - 1] - x[:, :, :img_height - 1, 1:], 2)
    return torch.sum(torch.pow(a + b, 1.25)) * tv_weight

def parse_torch_version():
    """
    Parses `torch.__version__` into a semver-ish version tuple.
    This is needed to handle subpatch `_n` parts outside of the semver spec.

    :returns: a tuple `(major, minor, patch, extra_stuff)`
    """
    match = re.match(r"(\d\.\d\.\d)(.*)", torch.__version__)
    major, minor, patch = map(int, match.group(1).split("."))
    extra_stuff = match.group(2)
    return major, minor, patch, extra_stuff 

        
def perlin(x,y,seed=0, use_numpy=True):
    def lerp(a,b,x):
        "linear interpolation"
        return a + x * (b-a)

    def fade(t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def gradient(h,x,y, use_numpy=True):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        vec_type = np.array if use_numpy else torch.cuda.FloatTensor
        vectors = vec_type([[0,1],[0,-1],[1,0],[-1,0]])
        g = vectors[h%4]
        return g[:,:,0] * x + g[:,:,1] * y
    
    device = torch.device("cuda")

    # permutation table
    np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    if use_numpy:
        xi = x.astype(int)
        yi = y.astype(int)
    else:
        p = torch.from_numpy(p).to(device)
        xi = torch.floor(x)
        yi = torch.floor(y)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)

    if not use_numpy:
        xi = xi.to(torch.long)
        yi = yi.to(torch.long)

    #ix = lambda x_1d, idx_2d: x_1d[idx_2d] if use_numpy else x_1d[idx_2d.view(-1)]
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf,use_numpy)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1, use_numpy)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1, use_numpy)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf, use_numpy)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u) 
    out = lerp(x1,x2,v) 

    return out

def add_noise(input_features, seed):
    device = torch.device("cuda")
    t0 = time.time()
    batch_size, channels, dim_y, dim_x = input_features.size()

    noisiness = 20

    use_numpy = True
    if use_numpy:
        lin_y = np.linspace(0, noisiness, dim_y, endpoint=False)
        lin_x = np.linspace(0, noisiness, dim_x, endpoint=False)
        x,y = np.meshgrid(lin_x,lin_y) 
    else:
        lin_y = torch.linspace(0, noisiness, dim_y, dtype=torch.float, device=device)
        lin_x = torch.linspace(0, noisiness, dim_x, dtype=torch.float, device=device)
        x,y = torch.meshgrid([lin_x,lin_y]) 

    # seed = np.random.randint(0,100)
    perlin_np = perlin(x,y,seed=seed,use_numpy=use_numpy)
    #print "perlin time: ", time.time() - t0
    t1 = time.time()
    perlin_t = torch.from_numpy(perlin_np).float().to(device)
    perlin_t = perlin_t.unsqueeze(0).expand(channels, -1, -1).unsqueeze(0)

    means = torch.mean(input_features[0,...].view(input_features.size(1),-1), dim=1)
    out = input_features * (perlin_t + 1.0)

    return out


def closestNumber(n, m): 
    # Find the quotient 
    q = int(n / m) 
    # 1st possible closest number 
    n1 = m * q 
    # 2nd possible closest number 
    if((n * m) > 0) : 
        n2 = (m * (q + 1))  
    else : 
        n2 = (m * (q - 1)) 
    # if true, then n1 is the required closest number 
    if (abs(n - n1) < abs(n - n2)) : 
        return n1 
    # else n2 is the required closest number  
    return n2 

def get_slider_vals(w,h, num_bins, upscale_fact):
    vals = []
    for i in range(w,w*upscale_fact+1):
        f = float(i)/w
        # both dims need to be int and
        # sizes need to be dividable by 4 to prevent downscaling pixel cutoffs
        if (h * f) - int(h * f) == 0.0 and (h * f) % 4 == i % 4 == 0:
            vals += [f]
            #print(f,w*f, h*f)

    if len(vals) > 2:
        ## quantize values into bins and take median 
        ## to get num_bins approx. equidistant vals
        bins = np.linspace(1.0,upscale_fact,num_bins)
        vals = np.array(vals)
        binned = np.digitize(vals, bins, right=False)
        slider_vals = [] 
        for _bin in range(1,num_bins+1):
            idx = np.argwhere(binned==_bin)
            if idx.size!=0:
                slider_vals += [vals[int(np.median(idx))]]

        if slider_vals[0] != 1.0:
            slider_vals = [1.0] + slider_vals
    elif len(vals) > 0:
        slider_vals = vals
    else:
        slider_vals = []
    return slider_vals


def get_files_sorted(search_dir):
    if not search_dir.endswith("/"):
        search_dir += "/"

    files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
    files.sort(key=lambda x: os.path.getmtime(x))

    return files


def construct_modelpath(module_path, module_subdir, dgf_type, style):
    model_dir = os.path.join(module_path, module_subdir, style + "_" + dgf_type)
    sfiles = get_files_sorted(model_dir)[::-1]
    models = list(filter(lambda s: s.endswith(".model"), sfiles))
    if len(models) >= 1:
        return models[0]
    
    checkpoint_models = list(filter(lambda s: s.endswith(".pth"), sfiles))
    if len(checkpoint_models) >= 1:
        return checkpoint_models[0]
    
    print("no model found in directory", model_dir)
    return None