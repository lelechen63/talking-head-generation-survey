# X2Face

We follow the official code https://github.com/oawiles/X2Face to implement work of [X2Face](http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/x2face.html).

## How to use

### Prerequest

- We use the code of py37_pytorch_0.4.1 branch
- Python = 3.7
- pytorch = 0.4.1
- For more detail, please check https://github.com/oawiles/X2Face/tree/py37_pytorch_0.4.1 .

### Dataset

Reference frame and target frames are required to run the model. Path to demo videos and target frame ids are stored in "vox_demo.npy", and videos are stored in directory "path". Both two demo videos are selected from voxceleb2 dataset.

### Evaluation

To evaluate with demo video, please run the following code:

```
python face2face_compare.py
```

Where number of reference images are set to 8.