import math
import os
import os.path
from PIL import Image
import numpy as np
import torch

from myutils import utils

from adjustable_upscaleNst import AdjustableNetwork
from adaptiveStrokeNet import JohnsonAdaptiveStroke
from reversibleEdit.reversible_rotation import ReversibleRotation
from reversibleEdit.reversible_warping import ReversibleWarping
from reversibleEdit.reversible_swirl import ReversibleSwirl
from reversibleEdit.reversible_warping import ReversibleWarpingSine
from reversibleEdit.vtk_tps import create_vtk_thin_spline_warp


def forward_backward_warp(warp_op, content_image, method):
    if method == 'optimize' or method == 'naive_reverse':
        content_image_warped = utils.tensor_load_rgbimage(content_image, size=1024, keep_asp=True, np_array=True, reversible_edit=warp_op.forward)
        content_image_original = utils.tensor_load_rgbimage(content_image, size=1024, keep_asp=True, np_array=True)
        warp_img = content_image_warped.transpose(1,2,0).astype('uint8')
        warp_img = Image.fromarray(warp_img)
        warp_img.save(f"output/warp_forward.jpg")


        if method == 'optimize':
            out_iters, grid = warp_op.backward_optimize(torch.from_numpy(content_image_warped).float().unsqueeze(0), 
                                                    torch.from_numpy(content_image_original).float().unsqueeze(0),  use_theta=True, use_src_as_dst=True)
            out_iters = out_iters.cpu().detach()[0].numpy().transpose(1,2,0).astype('uint8')
            out_iters = Image.fromarray(out_iters)
            out_iters.save(f"output/warp_iterative_backwards.jpg")
        else:
            out = warp_op.backward(warp_img)
            out.save(f"output/warp_naive_backwards.jpg")
    elif method == "vtk":
        img = np.array(Image.open(content_image)).astype('uint8') 
        warped_output = create_vtk_thin_spline_warp(img, warp_op.src_pts, warp_op.dst_pts , show_grid=False)
        warped_output = warped_output.transpose(2,0,1)
        Image.fromarray(warped_output).save(f"output/warp_vtk_forward.jpg" )
        warped_back = create_vtk_thin_spline_warp(warped_output, warp_op.src_pts, 
                                    warp_op.dst_pts, use_inverse=False, show_grid=False)

        warped_back = warped_back.transpose(2,0,1)
        Image.fromarray(warped_back).save(f"output/warp_vtk_backwards.jpg" )


def create_nst_adj_rot():
    CONTENT_IMAGE = "images/content/ferry.jpg"
    #STYLE = "sketch_girl"
    # STYLE = "abstractOrangePattern_2"
    STYLE = "mondrian"
    dir = "models/adjustable"

    REVERSIBLE_EDIT =  ReversibleRotation()# ReversibleSwirl()
    warping = True
    device = torch.device("cuda")
    model_path = [f.path for f in os.scandir(dir + "/" + STYLE) if f.name.endswith(".model")][0]

    def remove_IN_params(state_dict):
        new_dict = {}
        for (k,v) in state_dict.items():
            if not ".in1." in k:
                new_dict[k] = state_dict[k]
        return new_dict

    with torch.no_grad():
        style_model = AdjustableNetwork(fusion_krnsize=9)
        state_dict = remove_IN_params(torch.load(model_path))
        style_model.eval()
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        for i in range(0,360,1):
            print (i)
            REVERSIBLE_EDIT.rotation = i
            content_size = 1024 
            upscale_fact = 1
            warped_image = utils.tensor_load_rgbimage(CONTENT_IMAGE, size=1024, keep_asp=True, np_array=True, reversible_edit=REVERSIBLE_EDIT.forward)
            warped_image_small =  utils.tensor_load_rgbimage(CONTENT_IMAGE, size=int(content_size/upscale_fact), keep_asp=True, np_array=True, reversible_edit=REVERSIBLE_EDIT.forward)
            warped_image = torch.from_numpy(warped_image).float()
            warped_image = warped_image.unsqueeze(0).to(device)
            warped_image = utils.preprocess_batch(warped_image)

            warped_image_small = torch.from_numpy(warped_image_small).float()
            warped_image_small = warped_image_small.unsqueeze(0).to(device)
            warped_image_small = utils.preprocess_batch(warped_image_small)

            stroke_factor_input = torch.tensor([1.0]).to(device)
            output = style_model(warped_image_small, warped_image, stroke_factor_input).float().cpu().squeeze(0)
            img = output.clamp(0,255).numpy()
            img = img.transpose(1,2,0).astype('uint8')
            img = img[:,:,::-1]
            #output_warped_back = create_vtk_thin_spline_warp(img, REVERSIBLE_EDIT.src_pts, REVERSIBLE_EDIT.dst_pts, use_inverse=False, show_grid=False)
            out = REVERSIBLE_EDIT.backward(img)

            out.save(f"output/animation/rot_output_{i:04d}.jpg" )

    os.system(f"ffmpeg -i output/animation/rot_output_%04d.jpg -vcodec libx265 -crf 28 output/rot_output_-{STYLE}.mp4")

def create_nst_swirl():
    CONTENT_IMAGE = "images/content/ferry.jpg"
    OUT = "out.jpg"
    STYLE = "abstractOrangePattern_2"

    REVERSIBLE_EDIT =  ReversibleSwirl()
    warping = True
    device = torch.device("cuda")
    model_path = [f.path for f in os.scandir("models/adaptiveStroke/" + STYLE) if f.name.endswith(".model")][0]

    with torch.no_grad():
        style_model = JohnsonAdaptiveStroke()
        state_dict = torch.load(model_path)
        style_model.eval()
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        for i in range(80):
            print (i)
            REVERSIBLE_EDIT.strength = int(i / 4) if i <= 40 else 40 - int(i / 4)
            REVERSIBLE_EDIT.center = (560, 420)
            warped_image = utils.tensor_load_rgbimage(CONTENT_IMAGE, size=1024, keep_asp=True, np_array=True, reversible_edit=REVERSIBLE_EDIT.forward)

            warped_image = torch.from_numpy(warped_image).float()
            warped_image = warped_image.unsqueeze(0).to(device)
            warped_image = utils.preprocess_batch(warped_image)

            stroke_factor_input = torch.tensor([1.0]).to(device)
            output = style_model(warped_image, stroke_factor_input).float().cpu().squeeze(0)
            img = output.clamp(0,255).numpy()
            img = img.transpose(1,2,0).astype('uint8')
            img = img[:,:,::-1]
            out = REVERSIBLE_EDIT.backward(img)

            # output_warped_back = Image.fromarray(output_warped_back)
            out.save(f"output/animation/swirl_output_{i:04d}.jpg" )

    os.system(f"ffmpeg -i output/animation/swirl_output_%04d.jpg -vcodec libx265 -crf 28 output/swirl_output_-{STYLE}.mp4")

def create_nst_warping_animation():
    CONTENT_IMAGE = "images/content/ferry.jpg"
    STYLE = "delaunay"

    REVERSIBLE_EDIT = ReversibleWarping()
    device = torch.device("cuda")
    model_path = [f.path for f in os.scandir("models/adaptiveStroke/" + STYLE) if f.name.endswith(".model")][0]

    with torch.no_grad():
        style_model = JohnsonAdaptiveStroke()
        state_dict = torch.load(model_path)
        style_model.eval()
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        for i in range(80):
            print (i)
            REVERSIBLE_EDIT.dst_pts = torch.FloatTensor((
                # (0.45 - i * 0.001,0.5),(0.55 + i * 0.001,0.5),(0.5,0.45 - i * 0.001),(0.5,0.55 + i * 0.001)
                (0.45 ,0.5),(0.55 ,0.5),(0.5,0.45),(0.5,0.55 )
            )) 
            REVERSIBLE_EDIT.src_pts = torch.FloatTensor((
                (0.46- i * 0.001,0.5),(0.58 + i * 0.001,0.5),(0.5,0.43 - i * 0.001),(0.5,0.49 + i * 0.001)
            ))
            REVERSIBLE_EDIT.add_border_points(4)

            img = np.array(Image.open(CONTENT_IMAGE)).astype('uint8') 
            warped_image = create_vtk_thin_spline_warp(img, REVERSIBLE_EDIT.src_pts, REVERSIBLE_EDIT.dst_pts, show_grid=False)
            
            #warped_image = utils.tensor_load_rgbimage("output/animation/cur_forward.jpg", size=512, keep_asp=True) 
            warped_image = warped_image.transpose(2,0,1)
            warped_image = torch.from_numpy(warped_image).float()
            warped_image = warped_image.unsqueeze(0).to(device)
            warped_image = utils.preprocess_batch(warped_image)

            stroke_factor_input = torch.tensor([1.0]).to(device)
            output = style_model(warped_image, stroke_factor_input).float().cpu().squeeze(0)
            img = output.clamp(0,255).numpy()
            img = img.transpose(1,2,0).astype('uint8')
            img = img[:,:,::-1]
            output_warped_back = create_vtk_thin_spline_warp(img, REVERSIBLE_EDIT.src_pts, REVERSIBLE_EDIT.dst_pts, use_inverse=False, show_grid=False)

            output_warped_back = Image.fromarray(output_warped_back)
            output_warped_back.save(f"output/animation/output_{i:04d}.jpg" )

    os.system(f"ffmpeg -i output/animation/warped_nst_%04d.jpg -vcodec libx265 -crf 28 output/out-nst_warps-{STYLE}.mp4")
    os.system(f"ffmpeg -i output/animation/output_%04d.jpg -vcodec libx265 -crf 28 output/out-{STYLE}.mp4")

if __name__ == "__main__":
    #create_nst_warping_animation()
    create_nst_swirl()
    #nbb_warp_nst()
    #create_nst_warping_animation()
