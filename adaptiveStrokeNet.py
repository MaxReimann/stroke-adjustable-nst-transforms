# ============================================================
# PyTorch re-implementation of "Stroke Controllable Fast Style Transfer with Adaptive Receptive Fields", Jing et al., 2018
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
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from myutils import utils
import net.mynn as nn2
from myutils.vgg16 import Vgg16

from PIL import Image

STROKE_SHORTCUT_DICT = {"768": 2.0, "512": 1.0, "256": 0.0}
DEFAULT_RESOLUTIONS = ((768, 768), (512, 512), (256, 256))


######### config ##################
class Args(object):
    content_size = 256
    content_weight = 1.0
    style_weight = 10.0
    style_size = 512
    style_path = module_path + "/images/train_all_styles/wave.jpg"
    dataset = "/home/max/Datasets/mscoco" #"/projects/data/mscoco"
    batch_size = 8
    lr = 1e-3
    cuda = True
    epochs = 4
    log_interval = 100
    checkpoint_interval  = 10000
    checkpoint_model_dir = "models/adaptiveStroke/dreamstime_222964086"
    resume_train = None

    def __init__(self):
        if not os.path.exists(self.checkpoint_model_dir):
            os.makedirs(self.checkpoint_model_dir) 
    def setcheckpointdir(self, dir):
        self.checkpoint_model_dir = dir
        if not os.path.exists(self.checkpoint_model_dir):
            os.makedirs(self.checkpoint_model_dir) 

    def dump_config(self, model_name):
        attrs = [(attr,getattr(self, attr)) for attr in Args.__dict__.keys() if not attr.startswith('__')]
        out_str =  ',\n'.join("%s: %s" % item for item in attrs)
        with open(self.checkpoint_model_dir+"/train_config_{}.json".format(model_name.split(".")[0]),"w") as f:
            f.write("{\n" + out_str + "\n}")
##################################
args = Args()

class JohnsonAdaptiveStrokeDecoder(torch.nn.Module):
    def __init__(self):
        super(JohnsonAdaptiveStrokeDecoder, self).__init__()
        # Initial convolution layers
        self.conv1 = nn2.ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn2.ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn2.ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = nn2.ResidualBlock(128)
        self.res2 = nn2.ResidualBlock(128)
        self.res3 = nn2.ResidualBlock(128)
        self.res4 = nn2.ResidualBlock(128)
        self.res5 = nn2.ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = nn2.UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn2.UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = nn2.ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, y_res_combined):
        y = self.relu(self.in4(self.deconv1(y_res_combined)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class JohnsonAdaptiveStrokeEncoder(torch.nn.Module):
    def __init__(self):
        super(JohnsonAdaptiveStrokeEncoder, self).__init__()
        # Initial convolution layers
        self.conv1 = nn2.ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn2.ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn2.ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = nn2.ResidualBlock(128)
        self.res2 = nn2.ResidualBlock(128)
        self.res3 = nn2.ResidualBlock(128)
        self.res4 = nn2.ResidualBlock(128)
        self.res5 = nn2.ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = nn2.UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn2.UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = nn2.ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def _forward(self, X, stroke_factor):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y_res3 = self.res3(y)
        y_res4 = self.res4(y_res3)
        y_res5 = self.res5(y_res4)

        if stroke_factor <= 1.0:
            gamma = 0.0
            alpha = max(0.0, 1.0 - stroke_factor)
            beta = 1.0 - abs(stroke_factor - 1.0)
        else:
            stroke_factor -= 1
            alpha = 0.0
            beta = max(0.0, 1.0 - stroke_factor)
            gamma = 1.0 - abs(stroke_factor - 1.0)

        #print "alpha: {} beta: {} gamma: {}".format(alpha, beta, gamma)
        y_res_combined = alpha * y_res3 + beta * y_res4 + gamma * y_res5

        y = self.relu(self.in4(self.deconv1(y_res_combined)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y
    
    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y_res3 = self.res3(y)
        y_res4 = self.res4(y_res3)
        y_res5 = self.res5(y_res4)

        return (y_res3, y_res4, y_res5)

class JohnsonAdaptiveStroke(torch.nn.Module):
    def __init__(self):
        super(JohnsonAdaptiveStroke, self).__init__()
        # Initial convolution layers
        self.conv1 = nn2.ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn2.ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn2.ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = nn2.ResidualBlock(128)
        self.res2 = nn2.ResidualBlock(128)
        self.res3 = nn2.ResidualBlock(128)
        self.res4 = nn2.ResidualBlock(128)
        self.res5 = nn2.ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = nn2.UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn2.UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = nn2.ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X, stroke_factor):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y_res3 = self.res3(y)
        y_res4 = self.res4(y_res3)
        y_res5 = self.res5(y_res4)

        if stroke_factor <= 1.0:
            gamma = 0.0
            alpha = max(0.0, 1.0 - stroke_factor)
            beta = 1.0 - abs(stroke_factor - 1.0)
        else:
            stroke_factor -= 1
            alpha = 0.0
            beta = max(0.0, 1.0 - stroke_factor)
            gamma = 1.0 - abs(stroke_factor - 1.0)

        #print "alpha: {} beta: {} gamma: {}".format(alpha, beta, gamma)
        y_res_combined = alpha * y_res3 + beta * y_res4 + gamma * y_res5

        y = self.relu(self.in4(self.deconv1(y_res_combined)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y
    
    def _forward(self, X, alpha, beta, gamma):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y_res3 = self.res3(y)
        y_res4 = self.res4(y_res3)
        y_res5 = self.res5(y_res4)
    
        #print "alpha: {} beta: {} gamma: {}".format(alpha, beta, gamma)
        y_res_combined = alpha * y_res3 + beta * y_res4 + gamma * y_res5

        y = self.relu(self.in4(self.deconv1(y_res_combined)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y
  
############################# train ##############################################

def init_dataset(args):
    kwargs = {'num_workers': 2, 'pin_memory': False}
    transform = transforms.Compose([transforms.Resize(args.content_size),
                                    transforms.CenterCrop(args.content_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    print("reading dataset...")
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)
    print("finished reading dataset")

    return train_loader

def compute_style_gram(vgg, args, style_size):
    device = torch.device("cuda" if args.cuda else "cpu") 
    style_image = utils.tensor_load_rgbimage(args.style_path, size=style_size).to(device)
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
    style_model = JohnsonAdaptiveStroke().to(device)
    if args.resume_train is not None:
        print(('Resuming, initializing using weight from {}.'.format(args.resume_train)))
        style_model.load_state_dict(torch.load(args.resume_train))
    optimizer = Adam(style_model.parameters(), args.lr)

    vgg = Vgg16()
    utils.init_vgg16(module_path + "/models/")
    vgg.load_state_dict(torch.load(os.path.join(module_path,"models", "vgg16.weight")))
    vgg.to(device)

    style_grams = [compute_style_gram(vgg, args, size[0]) for size in DEFAULT_RESOLUTIONS]

    gc.collect()
    torch.cuda.empty_cache()


    ##### train loop #####

    DBG_PLOT = False
    if DBG_PLOT:
        import matplotlib
        import matplotlib.pyplot as plt
        plt.ion()
        plt.show()

    for e in range(args.epochs):
        style_model.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):  
            optimizer.zero_grad()
            n_batch = len(x)
            count += n_batch
            x = x.to(device)
            x = utils.preprocess_batch(x) # to BGR
            
            idx = batch_id % 3
            factor = STROKE_SHORTCUT_DICT[str(DEFAULT_RESOLUTIONS[idx][0])]
            #print "idx: {} factor: {} size: {}".format(idx, factor, str(DEFAULT_RESOLUTIONS[idx][0]))
            gram_style = style_grams[idx]


            y = style_model(x, factor)#, dbg_print=dbg_print)
            
            if DBG_PLOT and (batch_id + 1) % args.log_interval == 0 :
                img = y.data[0].clone().cpu().clamp(0,255).numpy().transpose(1, 2, 0).astype('uint8')
                # reverse bgr to rgb
                plt.imshow(img[:,:,::-1])
                plt.draw()
                plt.pause(0.01)

            y = utils.subtract_imagenet_mean_batch(y)
            features_y = vgg(y)

            xc = utils.subtract_imagenet_mean_batch(x)
            features_x = vgg(xc)
            
            content_loss = args.content_weight * mse_loss(features_y[1], features_x[1].detach())

            style_loss = 0
            for ft_y, gm_s, l in zip(features_y, gram_style, range(len(features_y))):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s.detach().expand(args.batch_size, -1, -1)[:n_batch, :, :])

            style_loss *= args.style_weight
            total_loss = content_loss + style_loss + utils.total_variation_loss(y, 1e-5) / args.batch_size
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/120000]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count,# len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                style_model.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_iter_" + str(batch_id * n_batch + 1) + \
                    str(time.ctime()).replace(' ', '_')  + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(style_model.state_dict(), ckpt_model_path)
                style_model.to(device).train()
                
    # save model
    style_model.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.checkpoint_model_dir, save_model_filename)
    args.dump_config(save_model_filename)
    torch.save(style_model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)