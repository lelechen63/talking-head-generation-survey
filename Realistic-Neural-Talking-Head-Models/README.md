# Realistic-Neural-Talking-Head-Models

For the work of Few-Shot Adversarial Learning of Realistic Neural Talking Head Models (Egor Zakharov et al.) (https://arxiv.org/abs/1905.08233), we apply the code implemented in https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models and modify some files to fit our work. The prerequisties VGGFace model,Libararies, pretrained weight, code architecture can be found in original website.

## How to use:

### Dataset

To run the model, reference images and landmarks are required. For demo videos, we store data path and targe frame ids in file "vox_demo.npy", which contains two videos from voxceleb2 dataset. Videos and landmarks can be found in directory "demo".

### Evaluation

We combine reference and fintune files in original code into "webcam_inferece_clean.py". To simply evaluate demo videos, you can run:

```
python webcam_inference_total.py
```

In the file, we set number of reference images to be 8, and finetune epoch to be 120.