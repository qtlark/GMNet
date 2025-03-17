<h1 align="center">[ICLR2025] Learning Gain Map for Inverse Tone Mapping</h1>

<p align="center">Yinuo Liao, Yuanshen Guan, Ruikang Xu, Jiacheng Li, Shida Sun, Zhiwei Xiong*</p>

<p align="center">
  <a href="https://openreview.net/pdf?id=GtHRhpgpzB" target="_blank">[Paper Link]</a>　
  <a href="#dataset">[Datasets]</a>　
  <a href="#code">[Codes]</a>　
  <a href="#script">[Scripts]</a>　
  <a href="#concact">[Contact]</a>
</p>

```tex
@inproceedings{Liao_2025_ICLR,
    title     = {Learning Gain Map for Inverse Tone Mapping},
    author    = {Yinuo Liao and Yuanshen Guan and Ruikang Xu and Jiacheng Li and Shida Sun and Zhiwei Xiong},
    booktitle = {The Thirteenth International Conference on Learning Representations},
    month     = {April},
    year      = {2025}
}
```



<h2 id="dataset">1. Datasets</h2>

We provide **Synthetic Dataset** and **Real-world Dataset**, which are organized by following four parts:

- `image`: The input SDR Images
- `gainmap`: The Gourd-Turth Gain Maps
- `metadata`: The metadata for restore HDR form SDR-GM pair (Only Qmax here)
- `thumbnail`: The down-sampled SDR Images in resolution `256×256` (Bicubic interpolation)

The data structure in dataset will be like:

```
synthetic_dataset
├── train
|   ├── image
|   |   └── *.png
|   ├── gainmap
|   |   └── *.png
|   ├── metadata
|   |   └── *.npy
|   └── thumbnail
|       └── *.png
└── test
    ├── image
    |   └── *.png
    ├── gainmap
    |   └── *.png
    ├── metadata
    |   └── *.npy
    └── thumbnail
        └── *.png
```

and more information can be found in the paper or the table below:

|             |      Synthetic Dataset      |     Real-world Dataset      |
| :---------: | :-------------------------: | :-------------------------: |
|   Source    |      HDR video frames       |        taken photos         |
|   Volume    | ㅤㅤㅤ900 trainset & 100 testsetㅤㅤㅤ | ㅤㅤㅤ900 trainset & 100 testsetㅤㅤㅤ |
| SDR White Level |           100 nit           |           203 nit           |
| HDR Peak Level |           800 nit           |          1015 nit           |
| ㅤㅤQmax Rangeㅤㅤ |     [0, 3]  ([0, log8])     |    [0, 2.32]  ([0,log5])    |
| Input SDR Image |                      3840×2160 8bit RGB                      |                      4096×3072 8bit RGB                      |
| Gourd-turth Gain Map |                     3840×2160 8bit Gray                      |                     2048×1536 8bit Gray                      |
| ㅤㅤDownload Linkㅤㅤ | [[BaiduNetDisk]](https://pan.baidu.com/s/1zIM6qP7KN1nQaQ0dFUg1Sg?pwd=1958) [[OneDrive]](https://www.zhihu.com/)　| [[BaiduNetDisk]](https://pan.baidu.com/s/1zIM6qP7KN1nQaQ0dFUg1Sg?pwd=1958) [[OneDrive]](https://www.zhihu.com/) |



<h2 id="code">2. Codes</h2>

### 2.1 How to test

Please download our dataset first, then modify the `dataroot` in `./codes/options/test/gmnet_test.yml`  to the path you store dataset, and you can modify `pretrain_model_G` to choose the pretrained model. When the configuration in `gmnet_test.yml` is ready, you can run the conmand:

```
cd codes
python test.py -opt options/config/test_syn.yml
```

The test results will be saved to `./results/test_name`.

### 2.2 How to train

To facilitate the training process, please modify the data path in `crop_training_patch.py` in <a href="#script">[Scripts]</a> and run it to crop the image and gainmap to patch:

```
cd scripts
python crop_training_patch.py --input_folder ../Synthetic_dataset/train/image --save_folder ../Synthetic_dataset/train/image_sp --n_thread 20 --crop_sz 480 --step 480
python crop_training_patch.py --input_folder ../Synthetic_dataset/train/gainmap --save_folder ../Synthetic_dataset/train/gainmap_sp --n_thread 20 --crop_sz 480 --step 480
```

It will generate pathes of `image` to `image_sp` folder, and the pathes of `gainmap` to `gainmap_sp` folder. After that, please modify the `dataroot` in `./codes/options/config/train_syn.yml`  to the sub-folder, then tun:

```
cd codes
python train.py -opt options/config/train_syn.yml
```

The checkpoints and training states can be found  `./experiments/train_name`.



<h2 id="script">3. Scripts</h2>

We provide several practical scripts in `./scripts` and the details are as following:

- `crop_training_patch.py`: This script crops the images to patches for training. (from [HDRTVNet](https://github.com/chxy95/HDRTVNet))
- `extract_double_layer_hdr.py`: The double-layer HDR image are store in one single file. This script extracts `sdr`, `gainmap` and `qmax` from double-layer file.
- `render_sdr_gm_to_linear_hdr.py`: This script restores linear HDR from `sdr`, `gainmap` and `qmax`.
- `pq_visualize.py`: This script converts the linear HDR to HDR image by PQ-OETF for visualization. 



<h2 id="concact">4. Contact</h2>

If you have any questions, please describe them in issues or contact yinuoliao@mail.ustc.edu.cn



<h2 id="ack">5. Acknowledgment</h2>

We appreciate the following github repositories for their valuable work:

- BasicSR: https://github.com/xinntao/BasicSR
- HDRSample: https://github.com/JonaNorman/HDRSample 
- HDR Toys: https://github.com/natural-harmonia-gropius/hdr-toys
