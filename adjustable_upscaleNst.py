# ============================================================
# Adjustable Style Transfer Architecture from "Interactive Multi-level Stroke Control for Neural Style Transfer, Reimann et al., 2021"
# Copyright 2022 Max Reimann
#
# Licensed under MIT License
# ============================================================
import os
import sys
import time
import numpy as np
import gc

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from myutils import utils
import net.mynn as nn2
from myutils.vgg16 import Vgg16

from PIL import Image
from tensorboardX import SummaryWriter
from guided_filter_pytorch.guided_filter import GuidedFilter#, ConvGuidedFilter
from net.guided_filter_module import DeepGuidedFilterConvGF, DeepGuidedFilterGuidedMapConvGF
if utils.parse_torch_version()[0] > 0 and utils.parse_torch_version()[1] < 3: #torch._thnn works only for up to 1.2
    from net.pacnet import PacConvTranspose2d

######### config ##################
class Args(object):
    content_size = 512
    content_weight = 1.0
    style_weight = 5.0
    style_size = 720
    style_path = module_path + "/images/train_all_styles/" + "les_apilles.jpg"
    dataset = "home/max/ml/mscoco-data/coco2017/images" # Dataset directory of training images, e.g. for MSCoco
    batch_size = 1
    lr = 1e-3
    cuda = True
    epochs = 2
    log_interval = 100
    precision="full"
    upscale_mode = "upconv" #upconv or pixelshuffle
    checkpoint_interval  = 10000
    checkpoint_model_dir = "models/"
    resume_train = None # Path of .pth if to continue training model
    pixel_kern_size = 9 # Kernel size of first convolution
    img_upscale_epoch2 = False
    # Type of guided upsampling module, or None for simple branch concatenation
    dgf_type = None #[None, "guided", "guidedConv", "guidedConvReversed", "guidedMapConv", "guidedMapConvReversed", "PAC" ]
    downscale_factors=[4, 2]
    viz_graph = True

    def __init__(self):
        if not os.path.exists(self.checkpoint_model_dir):
            os.makedirs(self.checkpoint_model_dir)

    def dump_config(self, model_name):
        attrs = [(attr,getattr(self, attr)) for attr in Args.__dict__.keys() if not attr.startswith('__')]
        out_str =  ',\n'.join("\"%s\": %s" % item for item in attrs)
        with open(self.checkpoint_model_dir+"/train_config_{}.json".format(model_name.split(".")[0]),"w") as f:
            f.write("{\n" + out_str + "\n}")
##################################
args = Args()

class AdjustableNetwork(torch.nn.Module):
    def __init__(self, dgf_type=None, upscale_mode="upconv", max_upscale=4, fusion_krnsize=3):
        super(AdjustableNetwork, self).__init__()

        self.style_branch = nn.Sequential(
            # Initial convolution layers
            nn2.ConvLayerCIN(3, 32, kernel_size=args.pixel_kern_size, stride=1), nn.ReLU(inplace=True),
            nn2.ConvLayerCIN(32, 64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn2.ConvLayerCIN(64, 128, kernel_size=3, stride=2), nn.ReLU(inplace=True),

            # Residual layers
            nn2.ResidualBlockCIN(128),
            nn2.ResidualBlockCIN(128),
            nn2.ResidualBlockCIN(128),
            nn2.ResidualBlockCIN(128),
            nn2.ResidualBlockCIN(128),
            nn2.ResidualBlockCIN(128),
            nn2.ResidualBlockCIN(128),

            # Upsampling Layers
            nn2.UpsampleConvLayerCIN(128, 64, kernel_size=3, stride=1, upsample=2), nn.ReLU(inplace=True),
            nn2.UpsampleConvLayerCIN(64, 32, kernel_size=3, stride=1, upsample=2), nn.ReLU(inplace=True),
        )

        self.dgf_type = dgf_type

        self.dynamic_upsample = DynamicUpsampleConvCIN(32, 32, kernel_size=3, upscale_mode=upscale_mode)
        self.hr_branch = nn.Sequential(
            nn2.ConvLayerCIN(3, 32, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn2.ConvLayerCIN(32, 32, kernel_size=3, stride=1), nn.ReLU(inplace=True)
        )
        
        self.fused_branch = nn.Sequential(
            nn2.ConvLayer(32 if self.dgf_type is not None else 32*2, 32, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn2.ConvLayer(32, 3, kernel_size=fusion_krnsize, stride=1)
        )

        # Fusion variants (deep guided filters) of ablation study
        if self.dgf_type is not None:
            if self.dgf_type == "guided":
                self.guided_filter = GuidedFilter(3, 1e-8)
            elif self.dgf_type == "guidedConv":
                self.guided_filter = DeepGuidedFilterConvGF(input_chans=32)
            elif self.dgf_type == "guidedMapConv":
                self.guided_filter = DeepGuidedFilterGuidedMapConvGF(input_chans=32)
            elif self.dgf_type == "guidedMapConvReversed":
                self.guided_filter = DeepGuidedFilterGuidedMapConvGF(input_chans=32, reversedParams=True)
            elif self.dgf_type == "guidedConvReversed":
                self.guided_filter = DeepGuidedFilterConvGF(input_chans=32, reversedParams=True)
            elif self.dgf_type == "PAC":
                self.dynamic_upsample = DynamicUpsampleConvCIN(32, 32, kernel_size=5, pac=True)

        self.param_predict = torch.nn.Sequential(nn.Linear(1, self._count_cin_params()))

        for m in self.param_predict.children():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.01)

    def to_precision(self, prec):
        if prec == "half":
            return self.half()
        else:
            return self.float()


    #convenience forward
    def forward(self, X, X_hr, style_weights, noise_seed=-1, add_unguided_factor=0.5, dbg_print=False):
        cin_params = self.param_predict(style_weights)
        split = int(cin_params.shape[0]/2)
        betas = cin_params[:split]
        gammas = cin_params[split:]

        return self._forward(X, X_hr, betas, gammas, noise_seed, add_unguided_factor, dbg_print)


    # forward for onnx converter
    def _forward(self, X, X_hr, betas, gammas, noise_seed=-1, add_unguided_factor=0.5, dbg_print=False):
        def run_with_IN_params(input, module_list, betas, gammas, param_iter, modify_feats=False):
            features_modified = False
            j,h = param_iter,input
            for module in module_list:
                if hasattr(module, 'num_cin_params'): 
                    j_new = j + int(module.num_cin_params / 2)
                    if modify_feats and not features_modified and type(module) == nn2.ResidualBlockCIN:
                        features_modified = True
                        if noise_seed > -1:
                            h = utils.add_noise(h, noise_seed)
                    h = module(h, betas[j:j_new], gammas[j:j_new])
                    if dbg_print:
                        print(type(module),  "(mean/var) betas: ({},{}) gammas: ({},{})".format(
                            torch.mean(betas[j:j_new]), torch.var(betas[j:j_new]), 
                            torch.mean(gammas[j:j_new]), torch.var(gammas[j:j_new])))
                    j = j_new
                else: 
                    h = module(h)
            return h,j

        h,j = run_with_IN_params(X, self.style_branch.children(), betas, gammas, param_iter=0, modify_feats=not self.training)
        target_size = X_hr.size()[-2:]

        if self.dgf_type != "PAC":
            j_new = int(j + self.dynamic_upsample.num_cin_params / 2)
            h_upsampled = F.relu(self.dynamic_upsample(h, target_size, betas[j:j_new], gammas[j:j_new]))
            y_hr,j = run_with_IN_params(X_hr, self.hr_branch.children(), betas, gammas, j_new)
        else:
            y_hr,j = run_with_IN_params(X_hr, self.hr_branch.children(), betas, gammas, j)
            j_new = int(j + self.dynamic_upsample.num_cin_params / 2)
            h_upsampled = F.relu(self.dynamic_upsample(h, target_size, betas[j:j_new], gammas[j:j_new], y_hr))


        if self.dgf_type is not None:
            if self.dgf_type.startswith("guided"):
                if self.dgf_type == "guidedConvReversed" or self.dgf_type == "guidedMapConvReversed":
                    h = F.interpolate(h, size=y_hr.size()[-2:], mode="bilinear")
                y = self.guided_filter(h, y_hr)
                y = y * ( 1.0 - add_unguided_factor) + h_upsampled * add_unguided_factor
            elif self.dgf_type == "PAC":
                y = h_upsampled
        else:
            y = torch.cat((h_upsampled,y_hr),1)

        return self.fused_branch(y)

    def _count_cin_params(self):
        count = 0
        layers = lambda module: list(module.children())
        for module in layers(self.style_branch) + [self.dynamic_upsample] + layers(self.hr_branch):
            if hasattr(module, 'num_cin_params'):
                count += module.num_cin_params
        return count


class DynamicUpsampling(nn.Module):
    def __init__(self):
        super(DynamicUpsampling, self).__init__()

    def forward(self, x, target):
        return F.interpolate(x, size=target , mode="bilinear")

class DynamicUpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DynamicUpsampleConv, self).__init__()
        self.dynamic_upsample_layer = DynamicUpsampling()
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)

    def forward(self, x, target_size):
        x = self.dynamic_upsample_layer(x, target_size)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out

class DynamicUpsamplePAC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DynamicUpsamplePAC, self).__init__()
        self.reflection_padding, op = int((kernel_size - 1) // 2), (kernel_size % 2)
        pad = int((kernel_size - 1) // 2)
        self.pac_upsample1 = PacConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=pad, output_padding=op)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=False)
        self.pac_upsample2 = PacConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=pad, output_padding=op)
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)

    def forward(self, x, guide, target_size):
        #if self.reflection_padding != 0:
        #    x = self.reflection_pad(x)
        if x.size(-1) * 2 >= target_size[-1]:
            x = F.interpolate(x, size=[f//2 for f in target_size], mode="bilinear")
            out = self.pac_upsample1(x, guide)
        elif x.size(-1) * 4 >= target_size[-1]:
            x = self.pac_upsample1(x, guide)
            x = F.relu(self.in1(x))
            if self.reflection_padding != 0:
                x = self.reflection_pad(x)
            x = self.pac_upsample2(x, guide)
            # downsample if needed
            out = F.interpolate(x, size=target_size, mode="bilinear")
        else:
            raise "Upsampling factor" + str(target_size[-1] / x.size(-1)) +  "not supported"
        return out

class DynamicUpsampleConvCIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pac=False, upscale_mode="upconv"):
        super(DynamicUpsampleConvCIN, self).__init__()
        self.pac = pac
        if pac:
            self.upsampling = DynamicUpsamplePAC(in_channels, out_channels, kernel_size)
        elif upscale_mode == "upconv":
            self.upsampling = DynamicUpsampleConv(in_channels, out_channels, kernel_size)
        elif upscale_mode == "pixelshuffle":
            self.upsampling = nn.PixelShuffle(4)

        self.in1 = nn2.ConditionalInstanceNorm(out_channels)
        self.num_cin_params = out_channels * 2

    def forward(self, x, target_size, beta, gamma, guide=None):
        if self.pac:
            out = self.upsampling(x, guide, target_size)
        else:
            out = self.upsampling(x, target_size)
        out = self.in1(out, beta, gamma)
        return out



############################# train ##############################################

class LHDDataset(datasets.ImageFolder):
    def __init__(self, dataset_path, content_size, downscale_factors=[2,4]):
        super(LHDDataset, self).__init__(dataset_path)
        self.content_size = content_size
        self.downscale_factors = downscale_factors
        
        self.hd_transform =  transforms.Compose([transforms.Resize(content_size),
                                transforms.CenterCrop(content_size),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.mul(255))])

        self.low_res_transforms = []
        for downscale_factor in self.downscale_factors:
            ld_transform =  transforms.Compose([transforms.Resize(int(content_size / downscale_factor)),
                                    transforms.CenterCrop(int(content_size / downscale_factor)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
            self.low_res_transforms.append(ld_transform)

    def __getitem__(self, index):
        img, _ = super(LHDDataset, self).__getitem__(index)
        return ([transform(img) for transform in self.low_res_transforms], self.hd_transform(img))


def init_dataset(args):
    kwargs = {'num_workers': 0, 'pin_memory': False}
    train_dataset = LHDDataset(args.dataset, args.content_size, downscale_factors=args.downscale_factors)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    return train_loader

def compute_style_gram(vgg, size):
    device = torch.device("cuda" if args.cuda else "cpu") 
    style_image = utils.tensor_load_rgbimage(args.style_path, size=size).to(device)
    style_image = style_image.unsqueeze(0)  
    style_image = utils.preprocess_batch(style_image)
    style_image =  utils.subtract_imagenet_mean_batch(style_image)
    features_style = vgg(style_image)
    gram_style = [utils.gram_matrix(y) for y in features_style]
    del features_style

    return gram_style

def train(args):
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if args.cuda else "cpu") 

    train_loader = init_dataset(args)

    mse_loss = torch.nn.MSELoss(reduction="elementwise_mean")
    style_model = AdjustableNetwork(args.dgf_type, max_upscale=max(args.downscale_factors)).to_precision(args.precision)
    if args.resume_train is not None:
        print(('Resuming, initializing using weight from {}.'.format(args.resume_train)))
        style_model.load_state_dict(torch.load(args.resume_train))
    

    writer = SummaryWriter(args.checkpoint_model_dir)
    
    if args.viz_graph:
        dummy_input = (torch.zeros(1, 3, int(args.content_size / args.downscale_factors[0]), int(args.content_size / args.downscale_factors[0])),
                                        torch.zeros(1, 3, args.content_size, args.content_size),                               torch.ones([1,]))
        writer.add_graph(style_model, dummy_input)

    style_model = style_model.to(device)
    optimizer = Adam(style_model.parameters(), args.lr)

    vgg = Vgg16()
    utils.init_vgg16(module_path + "/models/")
    vgg.load_state_dict(torch.load(os.path.join(module_path,"models", "vgg16.weight")))
    vgg.to(device)

    gc.collect()
    torch.cuda.empty_cache()

    sample_style_weight = torch.tensor([1.0]).to(device)
    args.dump_config("train_start")

    ##### train loop #####

    DBG_PLOT = True
    if DBG_PLOT:
        import matplotlib
        import matplotlib.pyplot as plt
        plt.ion()
        plt.show()

    for e in range(args.epochs):
        gram_styles = [compute_style_gram(vgg, int(args.style_size / f * 2)) for f in args.downscale_factors]
        style_model.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x_lowres_imgs, x_highres) in enumerate(train_loader):
            for x_lowres_index, x_lowres in enumerate(x_lowres_imgs):
                n_batch = len(x_lowres)
                count += n_batch
                optimizer.zero_grad()

                x_lowres = x_lowres.to(device)
                x_lowres = utils.preprocess_batch(x_lowres) # to BGR
                
                x_highres = x_highres.to(device)
                x_highres = utils.preprocess_batch(x_highres) # to BGR
                
                sample_style_weight.uniform_() # sample layer style weights from U(0,1)

                if args.precision == "half":
                    y = style_model(x_lowres.half(), x_highres.half(), sample_style_weight.half()).float()
                else:
                    y = style_model(x_lowres, x_highres, sample_style_weight)#, dbg_print=dbg_print)

                
                if DBG_PLOT and (batch_id + 1) % args.log_interval == 0 :
                    img = y.data[0].clone().cpu().clamp(0,255).numpy().transpose(1, 2, 0).astype('uint8')
                    # reverse bgr to rgb
                    plt.imshow(img[:,:,::-1])
                    plt.draw()
                    plt.pause(0.01)

                y = utils.subtract_imagenet_mean_batch(y)
                features_y = vgg(y)
                xc = utils.subtract_imagenet_mean_batch(x_highres)
                features_x = vgg(xc)
                
                content_loss = args.content_weight * mse_loss(features_y[1], features_x[1].detach())


                # style_losses = []
                style_loss = 0
                gram_style = gram_styles[x_lowres_index]
                for ft_y, gm_s, l in zip(features_y, gram_style, range(len(features_y))):
                    gm_y = utils.gram_matrix(ft_y)
                    style_loss += mse_loss(gm_y, gm_s.detach().expand(args.batch_size, -1, -1)[:n_batch, :, :]) * sample_style_weight


                style_loss *= args.style_weight
                total_loss = content_loss + style_loss
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()

                if args.img_upscale_epoch2:
                    log_now = e<2 and ((batch_id + 1) % args.log_interval == 0)
                    log_now = log_now or (e>=2 and ((batch_id + 1) % (args.log_interval * 4) == 0))
                else:
                    log_now = (batch_id + 1) % args.log_interval == 0
                
                if log_now:
                    mesg = "{}\tEpoch {}:\t[{}/120000]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count,# len(train_dataset),
                                      agg_content_loss / (batch_id + 1),
                                      agg_style_loss / (batch_id + 1),
                                      (agg_content_loss + agg_style_loss) / (batch_id + 1)
                    )
                    writer.add_scalar("Avg_Content_Loss", agg_content_loss / (batch_id + 1), batch_id*e + batch_id + 1)
                    writer.add_scalar("Avg_Style_Loss",  agg_style_loss / (batch_id + 1), batch_id*e + batch_id + 1)
                    writer.add_scalar("Avg_Total_Loss", (agg_content_loss + agg_style_loss) / (batch_id + 1),  batch_id*e + batch_id + 1)
                    print(mesg)
                    sys.stdout.flush()

                if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                    style_model.eval().cpu()
                    ckpt_model_filename = "ckpt_epoch_" + str(e) + "_iter_" + str(batch_id * n_batch + 1) + \
                        str(time.ctime()).replace(' ', '_')  + ".pth"
                    ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                    torch.save(style_model.state_dict(), ckpt_model_path)
                    style_model.to(device).train()
        if args.img_upscale_epoch2 and e == 1:
            args.batch_size = int(args.batch_size / 4)
            args.content_size *= 2
            args.style_size *= 2

            del gram_style, train_loader

            gc.collect()
            torch.cuda.empty_cache()

            train_loader = init_dataset(args)
            gram_style = compute_style_gram(vgg, args)
                
    # save model
    style_model.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.checkpoint_model_dir, save_model_filename)
    args.dump_config(save_model_filename)
    torch.save(style_model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


##################################### eval ##############################################

def stylize(content_image_path, size, model, style_weights, downscale):
    device = torch.device("cuda" if args.cuda else "cpu") 
    content_image = utils.tensor_load_rgbimage(content_image_path, size=int(size/downscale), keep_asp=True)#.half()
    content_image = content_image.unsqueeze(0).to(device)
    content_image = utils.preprocess_batch(content_image)

    content_image_HR = utils.tensor_load_rgbimage(content_image_path, size=size, keep_asp=True)#.half()
    content_image_HR = content_image_HR.unsqueeze(0).to(device)
    content_image_HR = utils.preprocess_batch(content_image_HR)
    
    
    with torch.no_grad():
        style_model = AdjustableNetwork()
        state_dict = torch.load(model)
        #style_model.half()
        style_model.eval()
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        output = style_model(content_image, content_image_HR, style_weights.uniform_()).float().cpu().squeeze(0)


        img = output.clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype('uint8')
        img = img[:,:,::-1]
        img = Image.fromarray(img)
    
    return img


#################################### main ####################################################
if __name__ == "__main__":
    method = "train"
    if method == "eval":
        model = args.checkpoint_model_dir + "/ckpt_epoch_0_iter_119997Sat_Dec_29_01:31:06_2018.pth"#
        content_image = module_path + "/images/content/mountain-chapel.jpg"
        size = 512
        style_weights = torch.Tensor([0.8,0.8,0.6,0.6]).to(torch.device("cuda")).unsqueeze(0).expand(4,-1)
        img_out_path = args.checkpoint_model_dir + "/hd_{0}px.jpg".format(size)

        img = stylize(content_image, size,  model, style_weights)
        img.save(img_out_path)
        print("saved output to: ", img_out_path)
    elif method == "train":
        train(args)