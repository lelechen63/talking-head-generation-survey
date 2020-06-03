import torch
import cv2
from matplotlib import pyplot as plt

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
# from webcam_demo.webcam_extraction_conversion import *
from webcam_demo.conversion_self import *

"""Init"""

#Paths
# path_to_model_weights = 'experiment/finetuned_model.tar'
path_to_model_weights = 'experiment/model_weights.tar'
path_to_embedding = 'examples/result/e_hat_video.tar'

device = torch.device("cuda:0")
cpu = torch.device("cpu")

checkpoint = torch.load(path_to_model_weights, map_location=cpu) 
e_hat = torch.load(path_to_embedding, map_location=cpu)
e_hat = e_hat['e_hat'].to(device)

G = Generator(256, finetuning=True, e_finetuning=e_hat)
G.eval()

"""Training Init"""
G.load_state_dict(checkpoint['G_state_dict'])
G.to(device)
G.finetuning_init()


"""Main"""
cap = cv2.VideoCapture("/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/01dfn2spqyE/00001_aligned.mp4")
# cap = cv2.VideoCapture("/home/cxu-serve/p1/common/voxceleb/test/video/mp4/id00812/1tr-i1_4O-A/00007.mp4")

with torch.no_grad():
    success, frame = cap.read()
    i = 0
    while success:
        print('finish {}'.format(i))
        x, g_y = generate_landmarks(frame=frame, device=device, pad=50)

        g_y = g_y.unsqueeze(0)
        x = x.unsqueeze(0)

        x_hat = G(g_y, e_hat)

        fig = plt.figure(figsize=(14, 14))
        # plt.clf()
        out1 = x_hat.transpose(1,3)[0]/255
        #for img_no in range(1,x_hat.shape[0]):
        #    out1 = torch.cat((out1, x_hat.transpose(1,3)[img_no]), dim = 1)
        out1 = out1.to(cpu).numpy()
        fig.add_subplot(1, 3, 1)
        plt.imshow(out1)
        # plt.savefig("examples/result/id00812/generate/ha.png")
        # plt.close()

        # plt.clf()
        out2 = x.transpose(1,3)[0]/255
        #for img_no in range(1,x.shape[0]):
        #    out2 = torch.cat((out2, x.transpose(1,3)[img_no]), dim = 1)
        out2 = out2.to(cpu).numpy()
        fig.add_subplot(1, 3, 2)
        plt.imshow(out2)
        # plt.savefig("examples/result/id00812/gt/%d5.png" % i)
        # plt.close()

        # plt.clf()
        out3 = g_y.transpose(1,3)[0]/255
        #for img_no in range(1,g_y.shape[0]):
        #    out3 = torch.cat((out3, g_y.transpose(1,3)[img_no]), dim = 1)
        out3 = out3.to(cpu).numpy()
        fig.add_subplot(1, 3, 3)
        plt.imshow(out3)
        # plt.savefig("examples/result/id00812/landmark/%d5.png" % i)
        # plt.close()

        plt.savefig("examples/result/self/total/%06d.jpg" % i)
        plt.close()

        # new frame
        i += 1
        success, frame = cap.read()

cap.release()
