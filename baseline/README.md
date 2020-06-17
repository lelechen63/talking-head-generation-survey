# Few-shot vid2vid \& Baseline

We run [few-shot-vid2vid](https://github.com/NVlabs/few-shot-vid2vid) following released official code https://github.com/NVlabs/few-shot-vid2vid. Moreover, we build baseline under this framework. Details of the two models will be introduced.

## Few-shot vid2vid

The prerequisities as well as training and testing process are the same as official code of [few-shot vid2vide](https://github.com/NVlabs/few-shot-vid2vid). We modify architecture of code for data loader and modules in order for better generality. But the model is kept as the same as [paper](https://github.com/NVlabs/few-shot-vid2vid). 

## Baseline

We adopt few-shot-vid2vid as backbone and optimize network structure to build baseline model. Specifically, Information Exchange module and ConvGate module are introduced to original attention part, as written in class "EncoderSelfAtten" of "generator_split.py". Moreover, we introduce Multi-branch Non-linear Combination module to combine warpped images with raw output of generator instead of image matting function as written in "architecture.py".

## Dataset

For dataloader, we use "facefore_dataset.py" for training and "facefore_demo_dataset.py" for evaluation. For both few-shot vid2vid and baseline, reference images and landmarks are required. Additionally, head rotation can be optionally used to guide the selection of image used for warpping, where rotation record of reference images are required.

## Training

To simply train the models with voxceleb2, run functions in "train_g8.sh". "train_vox_origin" can be used to train few-shot vid2vid, and "train_vox_new_nonlinear" can be used to train baseline. 

## Evaluation

To evaluate few-shot vid2vid, please run function "test_model_vox_origin" in "test_demo.sh". And to evaluate baseline, please run function "test_model_vox_new_noani".

## Important Flags

Besides flags mentioned in [few-shot-vid2vid](https://github.com/NVlabs/few-shot-vid2vid), several important noval flags are introduced here.

- Model
  - use_new: if specific, Information Exchange module, Conv

- Dataloader
  - no_rt: if specific, rotation record are not required in dataloader.
  - crop_ref: if specific, eyes and mouth regions of reference image are cropped, which are totally synthesized by generator.