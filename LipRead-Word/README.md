# LipRead-Word

This is a PyTorch implementation for lip reading in word level.

## Data

**LRS3 dataset:** We first copy all audio files and their corresponding transcripts to a single directory. Then we use Montreal Forced Aligner to generate word and phoneme level annotations. The instruction for it can be found [here](https://montreal-forced-aligner.readthedocs.io/en/stable/example.html). 

After that, we create a csv file (./repo/lrs3_word.csv) containing the top 300 words (without stop words) to be our entire dataset.

We crop the mouth region for each video sample. Please refer to [LipRead-seq2seq](https://github.com/arxrean/LipRead-seq2seq) for more details.

**VOX dataset:** Process the vox dataset with the same way as LRS3 dataset. Note that since VOX dataset has no sentence-level text, you should first generate texts based on audios.

### Train

Download [pretrained weights](https://drive.google.com/file/d/1vnO4QSgVRNutWPLLxi5oJfeoK6-YbA9g/view?usp=sharing) and move it to ./repo.

We train the model with lrs3 dataset.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pdb train.py --encode 233 --num_workers 16 --batch_size 256 --name lrs3 --gpu --gpus
```

Test the result on lrs3 test dataset. Uncomment *test(opt)* and comment *train(opt)* in *train.py*.

|       | Top1  | Top5  | Top10 |
| :---: | :---: | :---: | :---: |
| 50-w  |       |       |       |
| 100-w |       |       |       |
| 200-w | 76.31 | 91.16 | 93.57 |
| 300-w | 35.48 | 61.60 | 71.53 |

