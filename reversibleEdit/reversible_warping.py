from .reversible_edit import ReversibleEdit
import numpy as np
import torch


class ReversibleWarping(ReversibleEdit):
    def forward(self, image):
        return self.torch_warp(image)

    def torch_warp(self, image, reverse=False):
        pixels = np.asarray(image).transpose(2, 0, 1).copy()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        grid_y, grid_x = torch.meshgrid(torch.arange(0, pixels.shape[1], device=device), torch.arange(0, pixels.shape[2], device=device))
        grid_x, grid_y = self.warping_function(grid_x, grid_y, reverse)
        # grid_y = grid_y % pixels.shape[2] # uncomment for wrapping instead of reflection padding
        vgrid = torch.stack((grid_x, grid_y), 2).float().unsqueeze(0)  # W(x), H(y), 2
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(pixels.shape[1] - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(pixels.shape[2] - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = torch.nn.functional.grid_sample(torch.from_numpy(pixels).float().unsqueeze(0).to(device), vgrid_scaled, mode="bilinear", padding_mode="reflection")
        np_tensor = output.cpu().squeeze(0).detach().clamp(0, 255).numpy().astype('uint8')
        return np_tensor.transpose(1, 2, 0)

    def backward(self, image):
        return self.torch_warp(image, reverse=True)
        
    def warping_function(self, grid_x, grid_y, reverse=False):
        return grid_x, grid_y #Warping must be implemented by subclass


class ReversibleWarpingSine(ReversibleWarping):
    def __init__(self):
        self.period = 0.0
        self.stretch = 2
        self.amplitude = 25
        self.horizontal = False

    def forward(self, image):
        return self.torch_warp(image)

    def warping_function(self, grid_x, grid_y, reverse=False):
        grid = grid_y if self.horizontal else grid_x
        grid_out = grid_x if self.horizontal else grid_y
        offset = self.amplitude * torch.sin((self.stretch * 3.14 * grid / 180) + self.period * 2 * 3.14)
        if not reverse:
            grid_out = grid_out + offset
        else:
            grid_out = grid_out - offset
        return (grid_out, grid_y) if self.horizontal else (grid_x, grid_out)

    def backward(self, image):
        pixels = np.asarray(image)
        return self.torch_warp(image, reverse=True)