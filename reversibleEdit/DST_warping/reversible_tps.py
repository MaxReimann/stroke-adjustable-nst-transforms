from ..reversible_edit import ReversibleEdit
from PIL import Image
import numpy as np
import torch
from .warp import apply_warp
from . import tps
import torch.nn.functional as F


def to_numpy_image(x):
    return (x.detach().permute(0,2,3,1).numpy()*255).astype(np.uint8)

# reversible thin spline plate warping by. Works less well then VTK-based inversion
class ReversibleTPSOptim(ReversibleEdit):
    def __init__(self):
        self.src_pts = torch.FloatTensor((
            (0.5, 0.5),
        ))
        self.dst_pts = torch.FloatTensor((
            (0.5, 0.5),
        ))
        self.device = 'cpu'
        self.add_border_points()
        self.src = None


    def add_border_points(self, num_pts=50):
        w_pts = np.linspace(0,1,num=num_pts,endpoint=False)
        h_pts = np.linspace(0,1,num=num_pts,endpoint=False)

        border_pts = [[0, 0], [1, 0], [0, 1], [1, 1]]
        for i in range(1, num_pts):
            border_pts.append([h_pts[i], 0])
            border_pts.append([h_pts[i], 1])
            border_pts.append([0, w_pts[i]])
            border_pts.append([1, w_pts[i]])
        border_pts = torch.from_numpy(np.asarray(border_pts)).float()

        self.src_pts = torch.cat([self.src_pts, border_pts], 0)
        self.dst_pts = torch.cat([self.dst_pts, border_pts], 0)

    def forward(self, image):
        pixels = np.copy(np.asarray(image))
        in_img = torch.from_numpy(pixels).permute(2,0,1).unsqueeze(0).float()
        self.src = in_img.clone()
        im_warped = apply_warp(in_img, [self.src_pts], [self.dst_pts], self.device)
        im_warped = im_warped[0][0].permute(1,2,0).data.cpu().numpy()
        im_warped = np.clip(im_warped, 0, 255).astype(np.uint8)
        warped_image = Image.fromarray(im_warped)
        return warped_image

    def backward(self, image):
        pixels = np.copy(np.asarray(image))
        in_img = torch.from_numpy(pixels).permute(2,0,1).unsqueeze(0).float()
        im_warped = apply_warp(in_img, [self.dst_pts], [self.src_pts], self.device)
        im_warped = im_warped[0][0].permute(1,2,0).data.cpu().numpy()
        im_warped = np.clip(im_warped, 0, 255).astype(np.uint8)
        warped_image = Image.fromarray(im_warped)
        return warped_image

    def backward_optimize(self, image, target, use_theta=False, use_src_as_dst=False):
        src = image.cuda()
        target = target.cuda()
        c_dst = tps.uniform_grid((20,20)).view(-1, 2).to(target.device)

        if not use_src_as_dst:
            c_dst = tps.uniform_grid((20,20)).view(-1, 2).to(target.device)
        else:
            c_dst = self.src_pts.view(-1, 2).to(target.device)
        
        if not use_theta:
            theta = torch.zeros(1, (c_dst.shape[0]+2), 2).to(target.device)
            theta.requires_grad = True
        else:
            theta = (self.dst_pts - self.src_pts).to(target.device).unsqueeze(0)
            theta = torch.cat((theta,torch.zeros(1, 2, 2).to(target.device)), dim=1)
            theta.requires_grad = True
            
        size = src.shape
        opt = torch.optim.Adam([theta], lr=1e-2)
        for i in range(1000):
            opt.zero_grad()
            
            grid = tps.tps_grid(theta, torch.tensor(c_dst), size)
            warped = F.grid_sample(src, grid)

            loss = F.mse_loss(warped, target)
            loss.backward()
            opt.step()
            
            if i % 20 == 0:
                print(i, loss.item())

        return warped, grid