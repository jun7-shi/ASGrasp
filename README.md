# ASGrasp: Generalizable Transparent Object Reconstruction and 6-DoF Grasp Detection from RGB-D Active Stereo Camera
This repository contains the source code for our paper:
[ASGrasp: Generalizable Transparent Object Reconstruction and 6-DoF Grasp Detection from RGB-D Active Stereo Camera](https://arxiv.org/pdf/2405.05648.pdf)

For more information, please visit our [**project page**](https://pku-epic.github.io/ASGrasp/).

## Requirements
The code has been tested with `Python 3.8`, `PyTorch 1.9.1` and `Cuda 11.1`

## Installation
Please refer to https://github.com/rhett-chen/graspness_implementation to install GSNet

## Checkpoints
Please download the [checkpoints](https://drive.google.com/drive/folders/1omayRF-kl_HzkHRs9Ln7Dfn8L7dRUT9S?usp=sharing) to 'checkpoints/'

## Testing
Please run
```
python infer_mvs_2layer_gsnet.py
```

## Acknowledgement
Our code is based on these wonderful repos, we appreciate their great works!

* [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)
* [IterMVS](https://github.com/FangjinhuaWang/IterMVS)
* [GraspNet](https://github.com/graspnet)
* [graspness_implementation](https://github.com/rhett-chen/graspness_implementation)
* [DREDS](https://github.com/PKU-EPIC/DREDS)

## Contact
If you have any questions, please open a github issue or contact us:

Jun Shi: jun7.shi@samsung.com, Yong A: yong.a@samsung.com, He Wang: hewang@pku.edu.cn
