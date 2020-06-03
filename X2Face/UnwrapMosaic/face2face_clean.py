import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os, shutil
import glob
import torch
from PIL import Image
from torch.autograd import Variable
import torchvision
from torchvision.transforms import ToTensor, Compose, Scale
from torch.utils.data import DataLoader

from UnwrappedFace import UnwrappedFaceWeightedAverage, UnwrappedFaceWeightedAveragePose
from simple_grid import Grid, Vox

from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

import pdb

def main():
    # model
    BASE_MODEL = '../experiment/release_models/'
    state_dict = torch.load(BASE_MODEL + 'x2face_model_forpython3.pth')

    model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
    model.load_state_dict(state_dict['state_dict'])
    model = model.cuda()

    model = model.eval()

    # data
    save_path = "results/demo"
    # drive_file = ["./examples/Taylor_Swift/1.6/nuBaabkzzzI/"]
    # source_files = ["./examples/Taylor_Swift/1.6/nuBaabkzzzI/"]
    
    # source_root = "/home/cxu-serve/p1/common/voxceleb/test/img_sample"
    # source_files = []
    # for f in os.listdir(source_root):
    #     source_f = os.path.join(source_root, f)
    #     if len(os.listdir(source_f)) > 0:
    #         video_f = os.listdir(source_f)[0]
    #         img_f = os.listdir(os.path.join(source_f, video_f))[0]
    #         source_files.append(os.path.join(source_f, video_f, img_f))

    drive_file = ["/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id01567/cIZMA45dX0M/00291_aligned.mp4", 
                  "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/utfjXffHDgg/00198_aligned.mp4",
                  "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id01000/RvjbLfo3XDM/00052_aligned.mp4",
                  "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id04094/2sjuXzB2I1M/00025_aligned.mp4"]
    source_files = drive_file
    # drive_file = source_files

    # drive_data = Grid(drive_file, nums=10)
    # source_data = Grid(source_files, nums=10)

    drive_data = Vox(drive_file, nums=10)
    source_data = Vox(source_files, nums=8)

    drive_loader = DataLoader(dataset=drive_data, batch_size=1, shuffle=False)
    source_loader = DataLoader(dataset=source_data, batch_size=1, shuffle=False)

    # test
    with torch.no_grad():
        for d_index, drive_imgs in tqdm(drive_loader):
            drive_imgs = torch.cat(drive_imgs, dim=0)
            drive_imgs = drive_imgs.cuda()
            drive = torchvision.utils.make_grid(drive_imgs.cpu().data).permute(1,2,0).numpy()

            # source images
            for s_index, source_imgs in tqdm(source_loader):
                input_imgs = [img[0].repeat(drive_imgs.shape[0], 1, 1, 1) for img in source_imgs]
                # get image
                result = model(drive_imgs, *input_imgs)

                # store
                result = result.clamp(min=0, max=1)
                result_img = torchvision.utils.make_grid(result.cpu().data)
                result_img = result_img.permute(1,2,0).numpy()

                drive_file = drive_data.get_file(d_index.item()).split('/')[-2]
                file_name = os.path.join("{}.{}".format(*source_data.get_file(s_index.item()).split('/')[-2:]))
                file_name = os.path.join(save_path, drive_file, file_name)
                if not os.path.exists(file_name):
                    os.makedirs(file_name)
                else:
                    shutil.rmtree(file_name)
                    os.makedirs(file_name)

                plt.figure()
                plt.axis('off')
                plt.imshow(np.vstack((result_img, drive)))
                plt.savefig(os.path.join(file_name, "result.png"))
                plt.close()

                source_store = torchvision.utils.make_grid(torch.cat(source_imgs, dim=0).cpu().data)
                source_store = source_store.permute(1,2,0).numpy()
                plt.figure()
                plt.axis('off')
                plt.imshow(source_store)
                plt.savefig(os.path.join(file_name, "origin.png"))
                plt.close()

if __name__ == "__main__":
    main()