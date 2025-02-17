<h1 align="center">[ICLR2025] Learning Gain Map for Inverse Tone Mapping</h1>

<p align="center">Yinuo Liao, Yuanshen Guan, Ruikang Xu, Jiacheng Li, Shida Sun, Zhiwei Xiong*</p>

<p align="center">
  <a href="https://openreview.net/pdf?id=GtHRhpgpzB" target="_blank">[Paper Link]</a>　
  <a href="#code">[Codes]</a>　
  <a href="#script">[Scripts]</a>　
  <a href="#dataset">[Datasets]</a>　
  <a href="#concact">[Concact]</a>
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

<!-- <div align=center><img src="https://picx.zhimg.com/v2-3db8c51855060bd784b775e76e0100ea.png" style="width:70%" /></div> -->

<h2 id="code">1. Codes</h2>





<h2 id="script">2. Scripts</h2>





<h2 id="dataset">3. Datasets</h2>

|             |      Synthetic Dataset      |     Real-world Dataset      |
| :---------: | :-------------------------: | :-------------------------: |
|   Source    |      HDR video frames       |        taken photos         |
|   Volume    | 900 trainset & 100 testset  | 900 trainset & 100 testset  |
| White Level |           100 nit           |           203 nit           |
| Peak Level  |           800 nit           |          1015 nit           |
| ㅤㅤQmax Rangeㅤㅤ |     [0, 3]  ([0, log8])     |    [0, 2.32]  ([0,log5])    |
| Resolution  |          3840×2160          |          4096×3072          |
|    Depth    | 　ㅤㅤ8bit RGB SDR & 8bit Gray GMㅤㅤ　 | ㅤㅤ8bit RGB SDR & 8bit Gray GMㅤㅤ |



<h2 id="concact">4. Contact</h2>

If you have any questions, please submit issue or contact yinuoliao@mail.ustc.edu.cn



<h2 id="ack">5. Acknowledgment</h2>

We appreciate the following github repositories for their valuable work:

- BasicSR: https://github.com/xinntao/BasicSR
- HDRSample: https://github.com/JonaNorman/HDRSample 
- HDR Toys: https://github.com/natural-harmonia-gropius/hdr-toys
