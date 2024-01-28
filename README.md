# Computer Vision Final Project: Neural Style Transfer

梁世谦，裴虎镇，吴益强

This repository contains a PyTorch implementation of [Gatys' neural style transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), [LapStyle](https://arxiv.org/abs/1707.01253), [AdaIN](https://arxiv.org/abs/1703.06868), and AdaIN with Laplacian Loss.

Great thanks to the original authors, [this](https://d2l.ai/chapter_computer-vision/neural-style.html) tutorial and [this](https://github.com/naoto0804/pytorch-AdaIN) PyTorch implementation, both from which this implementation borrows code.

## Requirements

## Usage

### Gatys method

```bash
python gatys.py --method gatys --style path/to/style_img.jpg --content path/to/content_img.jpg
```

### LapStyle method

```bash
python gatys.py --method lapstyle --style path/to/style_img.jpg --content path/to/content_img.jpg
```

### AdaIN method

Adaptive instance normalization for arbitrary style transfer trains a feed-forward network instead of optimizing a noise image.

#### Download Models
```bash
bash get_vgg.sh
```
For a pre-trained decoder, we refer you to [this link](https://github.com/naoto0804/pytorch-AdaIN).

#### Test

```bash
python test_adain.py --style path/to/style_img.jpg --content path/to/content_img.jpg
```

#### Train
```bash
python train_adain.py --content_dir path/to/content/imgs --style_dir path/to/style/imgs
```

For more details and parameters, please refer to --help option.

## References
- [1]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.
- [2]: [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
