# P2LDGAN
- Official code for the "Joint Geometric-Semantic Driven Character Line Drawing Generation"(ICMR2023). [Link](https://doi.org/10.1145/3591106.3592216)

# News!
- Fortunately this article was accepted by ACM ICMR2023 and the Camera-ready version will be released soon.
- Our source code will be available before July 2023.

## Abstract
Character line drawing synthesis can be formulated as a special case of image-to-image translation problem that automatically manipulates the photo-to-line drawing style transformation. In this paper, we present the first generative adversarial network based end-to-end trainable translation architecture, dubbed P2LDGAN, for automatic generation of high-quality character drawings from input photos/images. The core component of our approach is the joint geometric-semantic driven generator, which uses our well-designed cross-scale dense skip connections framework to embed learned geometric and semantic information for generating delicate line drawings. In order to support the evaluation of our model, we release a new dataset including 1,532 well-matched pairs of freehand character line drawings as well as corresponding character images/photos, where these line drawings with diverse styles are manually drawn by skilled artists. Extensive experiments on our introduced dataset demonstrate the superior performance of our proposed models against the state-of-the-art approaches in terms of quantitative, qualitative and human evaluations.
<img src = 'imgs/network.jpg'>

## Pre-trained Models and Dataset
- You can download our pre-trained models which using P2LDGAN with our constructed line-drawing dataset via [Google Drive](https://drive.google.com/file/d/1To4V_Btc3QhCLBWZ0PdSNgC1cbm3isHP/view?usp=sharing).
- If you would like to use our dataset, you can send me an email indicating your name, organisation, purpose etc. and I will reply with a link to your dataset.

## Sample Results
(a) Input photo/image; (b) Ground truth; (c) Gatys; (d) CycleGAN; (e) DiscoGAN; (f) UNIT; (g) pix2pix; (h) MUNIT; (i) Our baseline; (j) Our P2LDGAN.
<img src = 'imgs/example.jpg'>
<img src = 'imgs/experiment.jpg'>

## Reference
If you use this work for a paper, please cite:

```
@inproceedings{fang2023p2ldgan,
author = {Fang, Cheng-Yu and Han, Xian-Feng},
title = {Joint Geometric-Semantic Driven Character Line Drawing Generation},
year = {2023},
isbn = {9798400701788},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3591106.3592216},
doi = {10.1145/3591106.3592216},
booktitle = {Proceedings of the 2023 ACM International Conference on Multimedia Retrieval},
pages = {226–233},
numpages = {8},
keywords = {Line Drawing, Joint Geometric-Semantic Driven, Generative Adversarial Network, Image Translation},
location = {Thessaloniki, Greece},
series = {ICMR '23}
}
```

