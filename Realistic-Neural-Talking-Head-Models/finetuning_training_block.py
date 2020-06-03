import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from datetime import datetime
from matplotlib import pyplot as plt


from dataset.dataset_class import FineTuningImagesDataset, FineTuningVideoDataset, FineTuningRefDataset
from network.model import *
from loss.loss_discriminator import *
from loss.loss_generator import *

import copy

"""Hyperparameters and config"""
def finetune(ref_imgs, ref_lmarks, path_to_chkpt=None, path_to_embedding=None, path_to_save=None, e_hat=None, num_epochs=40, ckpt=None):
    device = torch.device("cuda:0")
    cpu = torch.device("cpu")

    """Create dataset and net"""
    dataset = FineTuningRefDataset(ref_imgs, ref_lmarks, device)
    dataLoader = DataLoader(dataset, batch_size=2, shuffle=False)

    if path_to_embedding is not None:
        e_hat = torch.load(path_to_embedding, map_location=cpu)
        e_hat = e_hat['e_hat']
    else:
        assert e_hat is not None
        e_hat = e_hat.cpu()

    G = Generator(256, finetuning=True, e_finetuning = e_hat)
    D = Discriminator(dataset.__len__(), finetuning = True, e_finetuning = e_hat)

    G.train()
    D.train()

    optimizerG = optim.Adam(params = G.parameters(), lr=5e-5)
    optimizerD = optim.Adam(params = D.parameters(), lr=2e-4)


    """Criterion"""
    criterionG = LossGF(VGGFace_body_path='experiment/vggface/Pytorch_VGGFACE_IR.py',
                    VGGFace_weight_path='experiment/vggface/Pytorch_VGGFACE.pth', device=device)
    criterionDreal = LossDSCreal()
    criterionDfake = LossDSCfake()


    """Training init"""
    epoch = i_batch = 0
    lossesG = []
    lossesD = []

    #Warning if checkpoint inexistant
    if path_to_chkpt is not None:
        if not os.path.isfile(path_to_chkpt):
            print('ERROR: cannot find checkpoint')

        """Loading from past checkpoint"""
        checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    else:
        assert ckpt is not None
        checkpoint = copy.deepcopy(ckpt)
    checkpoint['D_state_dict']['W_i'] = torch.rand(512, dataset.__len__()) #change W_i for finetuning

    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])

    """Change to finetuning mode"""
    G.finetuning_init()
    D.finetuning_init()

    G.to(device)
    D.to(device)

    """Training"""
    batch_start = datetime.now()

    for epoch in range(num_epochs):
        for i_batch, (x, g_y) in enumerate(dataLoader):
            with torch.autograd.enable_grad():
                #zero the parameter gradients
                optimizerG.zero_grad()
                optimizerD.zero_grad()

                #forward
                #train G and D
                x_hat = G(g_y, e_hat)
                r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
                r, D_res_list = D(x, g_y, i=0)

                lossG = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list)
                lossDfake = criterionDfake(r_hat)
                lossDreal = criterionDreal(r)

                loss = lossDreal + lossDfake + lossG
                loss.backward(retain_graph=False)
                optimizerG.step()
                optimizerD.step()
                
                
                #train D again
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                x_hat.detach_()
                r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
                r, D_res_list = D(x, g_y, i=0)

                lossDfake = criterionDfake(r_hat)
                lossDreal = criterionDreal(r)

                lossD = lossDreal + lossDfake
                lossD.backward(retain_graph=False)
                optimizerD.step()


            # Output training stats
            if epoch % 10 == 0:
                batch_end = datetime.now()
                avg_time = (batch_end - batch_start) / 10
                print('\n\navg batch time for batch size of', x.shape[0],':',avg_time)
                
                batch_start = datetime.now()
                
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f'
                    % (epoch, num_epochs, i_batch, len(dataLoader),
                        lossD.item(), lossG.item(), r.mean(), r_hat.mean()))

            lossesD.append(lossD.item())
            lossesG.append(lossG.item())

    if path_to_save is not None:
        print('Saving finetuned model...')
        torch.save({
                'epoch': epoch,
                'lossesG': lossesG,
                'lossesD': lossesD,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                }, path_to_save)
        print('...Done saving latest')
    
    return G