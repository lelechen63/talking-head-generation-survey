import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import torch
from PIL import Image
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage, BottleneckFromNet
from sklearn.externals import joblib
from torchvision.transforms import Compose, Scale, ToTensor

import pdb

import warnings
warnings.simplefilter("ignore")

def load_img_and_audio(file_path):
    transform = Compose([Scale((256,256)), ToTensor()])
    img = Image.open(file_path).convert('RGB')
    img = transform(img)
    audio_label_path = str(file_path).replace('audio_faces', 'audio_features').replace('jpg','npz')
    audio_feature = torch.Tensor(np.load(audio_label_path)['audio_feat'])
    
    pdb.set_trace()

    return {'image' : img, 'audio' : audio_feature}

if __name__ == "__main__":
    # paths to source frames
    sourcepaths= ['examples/audio_faces/Retta/1.6/ALELNl9E1Jc/0002725.jpg',
                    'examples/audio_faces/Maya_Rudolph/1.6/Ylm6PVkbwhs/0004500.jpg', 
                'examples/audio_faces/Cristin_Milioti/1.6/IblJpk1GDZA/0004575.jpg',
                'examples/audio_faces/Peter_Capaldi/1.6/uAgUjSqIj7U/0001375.jpg']

    # path to frames corresponding to driving audio features
    audio_path = 'examples/audio_faces/Peter_Capaldi/1.6/uAgUjSqIj7U'
    imgpaths = os.listdir(audio_path)

    # loading models
    # BASE_MODEL = '/scratch/shared/slow/ow/eccv/2018/release_models/' # Change to your path
    BASE_MODEL = '../experiment/release_models/'
    # model_path = BASE_MODEL + 'x2face_model.pth'
    model_path = BASE_MODEL + "x2face_model_forpython3.pth"
    model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3,inner_nc=128)
    model.load_state_dict(torch.load(model_path)['state_dict'])

    s_dict = torch.load(model_path)
    modelfortargetpose = BottleneckFromNet()
    state = modelfortargetpose.state_dict()
    s_dict = {k: v for k, v in s_dict['state_dict'].items() if k in state.keys()}
    state.update(s_dict)
    modelfortargetpose.load_state_dict(state)

    posemodel = nn.Sequential(nn.Linear(128, 3))
    # p_dict_pre = torch.load(BASE_MODEL + '/posereg.pth')['state_dict']
    p_dict_pre = torch.load(BASE_MODEL + '/posereg_forpython3.pth')['state_dict']
    posemodel._modules['0'].weight.data = p_dict_pre['posefrombottle.weight'].cpu()
    posemodel._modules['0'].bias.data = p_dict_pre['posefrombottle.bias'].cpu()

    bottleneckmodel = nn.Sequential(nn.Linear(3, 128, bias=False), nn.BatchNorm1d(128))
    # b_dict_pre = torch.load(BASE_MODEL + '/posetobottle.pth')['state_dict']
    b_dict_pre = torch.load(BASE_MODEL + '/posetobottle_forpython3.pth')['state_dict']
    bottleneckmodel.load_state_dict(b_dict_pre)

    model = model.cuda()
    modelfortargetpose = modelfortargetpose.cuda()
    posemodel = posemodel.cuda()
    bottleneckmodel = bottleneckmodel.cuda()

    model.eval()
    modelfortargetpose.eval()
    posemodel.eval()
    bottleneckmodel.eval()

    # load linear regression from audio features to driving vector space
    linearregression = joblib.load(BASE_MODEL + '/linearregression_scaledTrue_7000.pkl')
    scalar = joblib.load(BASE_MODEL + '/scaler_7000.pkl')
    scalar = None


    # Drive 3 different identities with same audio
    img_gt_gen = np.empty((0,2560,3))
    for sourcepath in sourcepaths:
        img_to_show_all = np.empty((256,0,3))
        gt_ims = np.empty((256,0,3))
        source_data = load_img_and_audio(sourcepath)
        source_img = Variable(source_data['image']).cuda().unsqueeze(0)
        audio_feature_source = source_data['audio'].cpu().numpy().reshape(1,-1)
        audio_feature_origin = linearregression.predict(audio_feature_source)
        audio_feature_origin = torch.Tensor(audio_feature_origin).unsqueeze(2).unsqueeze(2)
        for imgpath in imgpaths:
            # Extract the driving audio features
            fullaudiopath = os.path.join(audio_path, imgpath)
            audio_data = load_img_and_audio(fullaudiopath)
            audio_img = Variable(audio_data['image'], volatile=True).cuda().unsqueeze(0)
            audio_feature = audio_data['audio'].cpu().numpy().reshape(1,-1)
            if not scalar is None:
                audio_feature = scalar.transform(audio_feature)
                audio_feature_origin = scalar.transform(audio_feature_origin)
            audio_feature = linearregression.predict(audio_feature)
            audio_feature = torch.Tensor(audio_feature).unsqueeze(2).unsqueeze(2)
            
            sourcebn = modelfortargetpose(source_img)
            sourcepose = posemodel(sourcebn.unsqueeze(0))
            sourceposebn = bottleneckmodel(sourcepose)
        
            def update_bottleneck(self, input, output):
                newdrive = sourcebn.unsqueeze(0).unsqueeze(2).unsqueeze(3) + Variable(audio_feature).cuda() - Variable(audio_feature_origin).cuda()
                audiopose =  posemodel(newdrive.squeeze().unsqueeze(0)) #
                audioposebn = bottleneckmodel(audiopose)
                output[0,:,:,:] = newdrive + sourceposebn.unsqueeze(2).unsqueeze(3) - audioposebn.unsqueeze(2).unsqueeze(3) # if we want to add old pose (of input) and substract pose info that's in the new bottleneck

            # Add a forward hook to update the model's bottleneck
            handle = model.pix2pixSampler.netG.model.submodule.submodule.submodule.submodule.submodule.submodule.submodule.down[1].register_forward_hook(update_bottleneck)
            result = model(source_img, source_img)
            handle.remove()
            img_to_show_all = np.hstack((result.squeeze().cpu().data.permute(1,2,0).numpy(), img_to_show_all))
            if img_gt_gen.shape == (0,2560,3):
                gt_ims = np.hstack((audio_img.squeeze().cpu().data.permute(1,2,0).numpy(), gt_ims))
        if img_gt_gen.shape == (0,2560,3):
            img_gt_gen = np.vstack((img_gt_gen, gt_ims))
        img_gt_gen = np.vstack((img_gt_gen, img_to_show_all))
    plt.rcParams["figure.figsize"] = [14,14]
    plt.imshow(img_gt_gen)
    plt.savefig("results/result.png")
    print('Top row: Frames corresponding to driving audio')
    print('Bottom 3 rows: generated frames driven with audio features corresponding to top row')