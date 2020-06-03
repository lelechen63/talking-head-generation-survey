import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os, shutil
import glob
import torch
from PIL import Image
from torch.autograd import Variable
import torchvision
import pickle as pkl
from torchvision.transforms import ToTensor, Compose, Scale
from torch.utils.data import DataLoader

from UnwrappedFace import UnwrappedFaceWeightedAverage, UnwrappedFaceWeightedAveragePose
from simple_grid import Grid, Vox, VoxSingle

from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

import pdb

# images to video
def image_to_video(sample_dir = None, video_name = None):
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%05d.jpg -c:v libx264 -y -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  ' + video_name 

    print (command)
    os.system(command)

def main():
    # model
    BASE_MODEL = '../experiment/release_models/'
    state_dict = torch.load(BASE_MODEL + 'x2face_model_forpython3.pth')

    model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
    model.load_state_dict(state_dict['state_dict'])
    model = model.cuda()

    model = model.eval()

    # data
    save_path = "extra_degree_result/vox"
    
    ref_files = []
    files = []
    
    pickle_files = np.load('vox_demo.npy', allow_pickle=True)
    files = list(set([f[0] for f in pickle_files]))
    tgt_ids = {}
    for f in pickle_files:
        if f[0] not in tgt_ids:
            tgt_ids[f[0]] = []
        tgt_ids[f[0]] += np.arange(int(f[1]), int(f[2])).tolist()
    ref_files = files

    # files = files[670:]
    total_f = len(files)
    files = [f for f in files if os.path.exists(f)]
    print('totally {} files while valid {} files'.format(total_f, len(files)))

    dataset = VoxSingle(files, ref_indx='0,5,10,15,20,25,30,35')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    # test
    state = 1
    with torch.no_grad():
        while state != -1:
            print("current {}/{}".format(state, len(files)))
            # get reference images
            ref_imgs = dataset.get_ref()
            # ref_imgs = ref_dataset.get_ref()
            # save reference images
            file = dataset.cur_file.split('/')
            path_name = '{}'.format(file[-1])
            save_file_name = os.path.join(save_path, path_name)
            for ref_id, img in enumerate(ref_imgs):
                save_ref_file = os.path.join(save_file_name, 'reference')
                if not os.path.exists(save_ref_file):
                    os.makedirs(save_ref_file)
                save_img = (img * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
                Image.fromarray(save_img).save(os.path.join(save_ref_file, '%05d.jpg'%ref_id))

            # preprocess
            ref_imgs = [ref.unsqueeze(0) for ref in ref_imgs]
            ref_imgs = torch.cat(ref_imgs, dim=0)
            ref_imgs = ref_imgs.cuda()
            ref = torchvision.utils.make_grid(ref_imgs.cpu().data).permute(1,2,0).numpy()
            # synthesize
            for d_index, drive_img in tqdm(dataloader):
                if d_index not in tgt_ids[dataset.cur_file]:
                    continue
                if os.path.exists(os.path.join(save_file_name, '%05d.jpg'%d_index)):
                    continue

                drive_img = drive_img.cuda()

                input_imgs = [img.repeat(drive_img.shape[0], 1, 1, 1) for img in ref_imgs]
                # get image
                result = model(drive_img, *input_imgs)

                # store
                result = result.clamp(min=0, max=1)
                result_img = torchvision.utils.make_grid(result.cpu().data)
                result_img = result_img.permute(1,2,0).numpy()

                save_img = drive_img.clamp(min=0, max=1)
                save_img = torchvision.utils.make_grid(save_img.cpu().data)
                save_img = save_img.permute(1,2,0).numpy()

                # final_img = np.hstack([result_img, save_img])
                final_img = result_img

                final_img = (final_img * 255).astype(np.uint8)
                Image.fromarray(final_img).save(os.path.join(save_file_name, '%05d.jpg'%d_index))

            # combine video
            # image_to_video(save_file_name, os.path.join(save_file_name, '{}.mp4'.format(path_name)))

            # new file
            state = dataset.nextfile()


if __name__ == "__main__":
    main()