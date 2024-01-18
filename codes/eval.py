import torch
import torch.nn as nn
import os
import argparse

from dataset import dataloader as db
from network import get_model
from utils import *
from math import log10
import pytorch_ssim
from argp import args

from collections import OrderedDict
from runpy import run_path
import configparser
from EDSRmodel.edsr import EDSR
import CARM
import time



def run_eval(args):
    netG1 = CARM().cuda()
    netG2 = EDSR(args).cuda()  # SR

    model = nn.ModuleList()

    model.append(netG1)  # DN
    model.append(netG2)  # LR

   
    # checkpoint = torch.load('./pretrain_model/main20_1st_epoch3.pth')
    # checkpoint = torch.load('./pretrain_model/main20_2nd_epoch1.pth')
    # checkpoint = torch.load('./checkpoint/main20_2nd/model_epoch3_600.pth')
    checkpoint = torch.load('./pretrain_model/main20_3rd_epoch3_600.pth')
    # checkpoint = torch.load('./pretrain_model/main20_4th_epoch4_2400.pth')
    
    model.load_state_dict(checkpoint["model"].state_dict()) 
    model.eval()
    

    up_sampler = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True).cuda()

    video_loader = db.HRLRDLR_VideoLoadertest(args, args.test_path, video_size=(args.size[0],args.size[1]), train = False)
    print('Number of samples: {}'.format(len(video_loader)))

    criterionMSE = nn.MSELoss().cuda()
    counter = 0

    seq_config = configparser.ConfigParser()
    seq_config.read(args.test_size)

    with torch.no_grad():
        for i in range(len(video_loader)):
            data_sample, _vname = video_loader.get_item(i)
            print('Processing ' + _vname)

            DLR_frames = data_sample['DLR']
            LR_frames = data_sample['LR']
            HR_frames = data_sample['HR']

            yuv_name = _vname+'.yuv'
            width = seq_config[yuv_name].getint('width')
            height = seq_config[yuv_name].getint('height')

            

            hr_writer = YUVWriter(os.path.join(args.out, _vname + '_HR.yuv'), (height*2, width*2))
            # lr_writer = YUVWriter(os.path.join(args.out, _vname + '_LR.yuv'), (height, width))

            errors = AverageMeters(args, _vname, ['BICUBIC-HR-Y-PSNR', 'HR-Y-PSNR', 'DLR-Y-PSNR',  'NET-LR-Y-PSNR', 'Coding_Time'])

            for frame_id in range(len(DLR_frames)):
                # Init data
                DLR = torch.autograd.Variable(torch.from_numpy(DLR_frames[frame_id].transpose((2, 0, 1))).float()).unsqueeze(0).cuda() / 255.0
                LR = torch.autograd.Variable(torch.from_numpy(LR_frames[frame_id].transpose((2, 0, 1))).float()).unsqueeze(0).cuda() / 255.0
                HR = torch.autograd.Variable(torch.from_numpy(HR_frames[frame_id].transpose((2, 0, 1))).float()).unsqueeze(0).cuda() / 255.0

                # Inference
                DLRY = DLR[:, :1, :, :]
                LRY = LR[:, :1, :, :]
                HRY = HR[:, :1, :, :]
                DLRY3 = torch.cat(([DLRY] * 3), dim=1)
                LRY3 = torch.cat(([LRY] * 3), dim=1)
                HRY3 = torch.cat(([HRY] * 3), dim=1)

                torch.cuda.synchronize() #增加同步操作
                start1 = time.time()
                net_LRY3 = model[0](DLRY3)[1]
                torch.cuda.synchronize() #增加同步操作
                end1 = time.time()
                LR_out = torch.cat([net_LRY3[:,:1,:,:], DLR[:,1:,:,:]], 1)

                bicubic_HR = torch.cat([up_sampler(net_LRY3[:, :1, :, :]) , up_sampler(DLR[:, 1:, :, :])], 1)

                torch.cuda.synchronize() #增加同步操作
                start2 = time.time()
                net_HRY3 = model[1](net_LRY3 * 255.0) / 255.0
                torch.cuda.synchronize() #增加同步操作
                end2 = time.time()
                net_HR = torch.cat([net_HRY3[:, :1, :, :], up_sampler(DLR[:, 1:, :, :])], 1)

                coding_time = end2  - start1

                
                # Evaluation
                dlr_y_mse = criterionMSE((DLR[:,0,:,:].unsqueeze(1)* 255.0 + 0.5).int().float(), (LR[:,0,:,:].unsqueeze(1)* 255.0 + 0.5).int().float())
                dlr_y_psnr = 10 * log10(255.0*255.0 / dlr_y_mse.item())
                # dlr_y_ssim = pytorch_ssim.ssim(DLR[:,0,:,:].unsqueeze(1), LR[:,0,:,:].unsqueeze(1)).item()

                lr_y_mse = criterionMSE((LR_out[:,0,:,:].unsqueeze(1)* 255.0 + 0.5).int().float(), (LR[:,0,:,:].unsqueeze(1)* 255.0 + 0.5).int().float())
                lr_y_psnr = 10 * log10(255.0*255.0 / lr_y_mse.item())
                # lr_y_ssim = pytorch_ssim.ssim(LR_out[:,0,:,:].unsqueeze(1), LR[:,0,:,:].unsqueeze(1)).item()

                bicubic_hr_y_mse = criterionMSE((bicubic_HR[:, 0, :, :].unsqueeze(1) * 255.0 + 0.5).int().float(), 
                                        (HR[:, 0, :, :].unsqueeze(1) * 255.0 + 0.5).int().float())  
                bicubic_hr_y_psnr = 10 * log10(255.0 * 255.0 / bicubic_hr_y_mse.item())
                hr_y_mse = criterionMSE((net_HR[:, 0, :, :].unsqueeze(1) * 255.0 + 0.5).int().float(), (
                                    HR[:, 0, :, :].unsqueeze(
                                        1) * 255.0 + 0.5).int().float())  # net's out_tensor -> int -> float -> MSE
                hr_y_psnr = 10 * log10(255.0 * 255.0 / hr_y_mse.item())


                
                errors.update([
                     bicubic_hr_y_psnr, hr_y_psnr, dlr_y_psnr, lr_y_psnr, coding_time
                ])

                # CPU
                # res_residual_out = res_residual_out.cpu()
                # rec_residual_out = rec_residual_out.cpu()
                # upsampled_lr = upsampled_lr.cpu()
                # HR_out = HR_out.cpu()
                DLR = DLR.cpu()
                LR = LR.cpu()
                HR = HR.cpu()

                # Write frames
                hr_writer.write(net_HR)
                # lr_writer.write(LR_out)
            # Finish
            hr_writer.close()
            # lr_writer.close()

            errors.write_result()

    print('Test Completed')

if __name__ == '__main__':

    run_eval(args)
