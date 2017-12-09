## Introduction

Codes for reproducing [Associative Embedding: End-to-End Learning for Joint Detection and Grouping](https://arxiv.org/abs/1611.05424)

## Results

<div align='center'>
<img src="https://github.com/JakeRenn/Pytorch-Multi-Person-Pose-Estimation/blob/master/media/ori_images/ski.jpg", width="300", height="300">
&nbsp;
<img src="https://github.com/JakeRenn/Pytorch-Multi-Person-Pose-Estimation/blob/master/pics/ski.jpg", width="300", height="300">
&nbsp;
<img src="https://github.com/JakeRenn/Pytorch-Multi-Person-Pose-Estimation/blob/master/pics/leaderboard.png", width="300", height="300">
</div>
</div>

## Contents

1. src: source codes including model, data reader and utils
2. tools: main functions to run testing or visualizing
3. scripts: scripts to run testing or visualizing

Other directories are self-explanatory

## Require

* Python2.7
* Opencv
* Pytorch
* CUDNN
* numpy

## Instructions

Pretrained model is available [here](https://pan.baidu.com/s/1nvKJlFz). Include the model in the `./checkpoints` directory or modify the variable `CHECKPOINT` in `./scripts/test.sh`.


Run
```
./scripts/init_dir.sh
```
to make necessary directories.

Run
```
./scripts/test.sh
```
to test model on images in `media` directory. Or you may change the variable `IMAGES_DIR` in `./scripts/test.sh` to test on your own images.

In the same way, run
```
./scripts/visualize.sh
```
to visualize results. The rendered images will be save at `./media/pre_images`

## Output Format
In the JSON format, results are as follow

```json
[
    {
        "image_id": "a0f6bdc065a602b7b84a67fb8d14ce403d902e0d",
        "keypoint_annotations": {
        "human1": [261, 294, 1, 281, 328, 1, 0, 0, 0, 213, 295, 1, 208, 346, 1, 192, 335, 1, 245, 375, 1, 255, 432, 1, 244, 494, 1, 221, 379, 1, 219, 442, 1, 226, 491, 1, 226, 256, 1, 231, 284, 1],
        "human2": [313, 301, 1, 305, 337, 1, 321, 345, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 313, 359, 1, 320, 409, 1, 311, 454, 1, 0, 0, 0, 330, 409, 1, 324, 446, 1, 337, 284, 1, 327, 302, 1],
        "human3": [373, 304, 1, 346, 286, 1, 332, 263, 1, 0, 0, 0, 0, 0, 0, 345, 313, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 363, 386, 1, 361, 424, 1, 361, 475, 1, 365, 273, 1, 369, 297, 1],
        }
    }
]
```

Detailed explanation could be found in the website [AI Challenger](https://challenger.ai/competition/keypoint/subject)

## Notice

* The time to release training codes is not decided yet.
* The model is simplified due to GPU memory limitation. The output feature map is 8 times smaller than the input image instead.

