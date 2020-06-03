import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import torch
from PIL import Image
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage, UnwrappedFaceWeightedAveragePose
import torchvision
from torchvision.transforms import ToTensor, Compose, Scale

def run_batch(source_images, pose_images):
    return model(pose_images, *source_images)

BASE_MODEL = '../experiment/release_models/' # Change to your path
state_dict = torch.load(BASE_MODEL + 'x2face_model_forpython3.pth')

model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
model.load_state_dict(state_dict['state_dict'])
model = model.cuda()

model = model.eval()

driver_path = './examples/Taylor_Swift/1.6/nuBaabkzzzI/'
source_path = './examples/Taylor_Swift/1.6/vBgiDYBCuxY/'

driver_imgs = [driver_path + d for d in sorted(os.listdir(driver_path))][0:8] # 8 driving frames
source_imgs  = [source_path + d for d in sorted(os.listdir(source_path))][0:3] # 3 source frames

def load_img(file_path):
    img = Image.open(file_path)
    transform = Compose([Scale((256,256)), ToTensor()])
    return Variable(transform(img)).cuda()

# Driving the source image with the driving sequence
source_images = []
for img in source_imgs:
    source_images.append(load_img(img).unsqueeze(0).repeat(len(driver_imgs), 1, 1, 1))
    
driver_images = None
for img in driver_imgs:
    if driver_images is None:
        driver_images = load_img(img).unsqueeze(0)
    else:
        driver_images = torch.cat((driver_images, load_img(img).unsqueeze(0)), 0)

# Run the model for each
with torch.no_grad():
    result = run_batch(source_images, driver_images)
result = result.clamp(min=0, max=1)
img = torchvision.utils.make_grid(result.cpu().data)
    
# Visualise the results
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 24.
fig_size[1] = 24.
plt.rcParams["figure.figsize"] = fig_size
plt.axis('off')

result_images = img.permute(1,2,0).numpy()
driving_images = torchvision.utils.make_grid(driver_images.cpu().data).permute(1,2,0).numpy()
plt.imshow(np.vstack((result_images, driving_images)))

plt.savefig(os.path.join('../results/demo', 'result.png'))
plt.close()