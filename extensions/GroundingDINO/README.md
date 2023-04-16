# Grounding DINO 

---

[![arXiv](https://img.shields.io/badge/arXiv-2303.05499-b31b1b.svg)](https://arxiv.org/abs/2303.05499) 
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/wxWDt5UiwY8)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/cMa77r3YrDk)
[![HuggingFace space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/ShilongLiu/Grounding_DINO_demo)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/zero-shot-object-detection-on-mscoco)](https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco?p=grounding-dino-marrying-dino-with-grounded) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/zero-shot-object-detection-on-odinw)](https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw?p=grounding-dino-marrying-dino-with-grounded) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=grounding-dino-marrying-dino-with-grounded) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=grounding-dino-marrying-dino-with-grounded)



Official PyTorch implementation of [Grounding DINO](https://arxiv.org/abs/2303.05499), a stronger open-set object detector. Code is available now!


## Highlight

- **Open-Set Detection.** Detect **everything** with language!
- **High Performancce.** COCO zero-shot **52.5 AP** (training without COCO data!). COCO fine-tune **63.0 AP**.
- **Flexible.** Collaboration with Stable Diffusion for Image Editting.

## News
[2023/03/28] A YouTube [video](https://youtu.be/cMa77r3YrDk) about Grounding DINO and basic object detection prompt engineering. [[SkalskiP](https://github.com/SkalskiP)] \
[2023/03/28] Add a [demo](https://huggingface.co/spaces/ShilongLiu/Grounding_DINO_demo) on Hugging Face Space! \
[2023/03/27] Support CPU-only mode. Now the model can run on machines without GPUs.\
[2023/03/25] A [demo](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb) for Grounding DINO is available at Colab. [[SkalskiP](https://github.com/SkalskiP)] \
[2023/03/22] Code is available Now!

<details open>
<summary><font size="4">
Description
</font></summary>
<img src=".asset/hero_figure.png" alt="ODinW" width="100%">
</details>



## TODO 

- [x] Release inference code and demo.
- [x] Release checkpoints.
- [ ] Grounding DINO with Stable Diffusion and GLIGEN demos.
- [ ] Release training codes.

## Install 

If you have a CUDA environment, please make sure the environment variable `CUDA_HOME` is set. It will be compiled under CPU-only mode if no CUDA available.

```bash
pip install -e .
```

## Demo

```bash
CUDA_VISIBLE_DEVICES=6 python demo/inference_on_a_image.py \
  -c /path/to/config \
  -p /path/to/checkpoint \
  -i .asset/cats.png \
  -o "outputs/0" \
  -t "cat ear." \
  [--cpu-only] # open it for cpu mode
```
See the `demo/inference_on_a_image.py` for more details.

**Web UI**

We also provide a demo code to integrate Grounding DINO with Gradio Web UI. See the file `demo/gradio_app.py` for more details.

## Checkpoints

<!-- insert a table -->
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>Data</th>
      <th>box AP on COCO</th>
      <th>Checkpoint</th>
      <th>Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>GroundingDINO-T</td>
      <td>Swin-T</td>
      <td>O365,GoldG,Cap4M</td>
      <td>48.4 (zero-shot) / 57.2 (fine-tune)</td>
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth">Github link</a> | <a href="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth">HF link</a></td>
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py">link</a></td>
    </tr>
  </tbody>
</table>

## Results

<details open>
<summary><font size="4">
COCO Object Detection Results
</font></summary>
<img src=".asset/COCO.png" alt="COCO" width="100%">
</details>

<details open>
<summary><font size="4">
ODinW Object Detection Results
</font></summary>
<img src=".asset/ODinW.png" alt="ODinW" width="100%">
</details>

<details open>
<summary><font size="4">
Marrying Grounding DINO with <a href="https://github.com/Stability-AI/StableDiffusion">Stable Diffusion</a> for Image Editing
</font></summary>
<img src=".asset/GD_SD.png" alt="GD_SD" width="100%">
</details>

<details open>
<summary><font size="4">
Marrying Grounding DINO with <a href="https://github.com/gligen/GLIGEN">GLIGEN</a> for more Detailed Image Editing
</font></summary>
<img src=".asset/GD_GLIGEN.png" alt="GD_GLIGEN" width="100%">
</details>

## Model

Includes: a text backbone, an image backbone, a feature enhancer, a language-guided query selection, and a cross-modality decoder.

![arch](.asset/arch.png)


## Acknowledgement

Our model is related to [DINO](https://github.com/IDEA-Research/DINO) and [GLIP](https://github.com/microsoft/GLIP). Thanks for their great work!

We also thank great previous work including DETR, Deformable DETR, SMCA, Conditional DETR, Anchor DETR, Dynamic DETR, DAB-DETR, DN-DETR, etc. More related work are available at [Awesome Detection Transformer](https://github.com/IDEACVR/awesome-detection-transformer). A new toolbox [detrex](https://github.com/IDEA-Research/detrex) is available as well.

Thanks [Stable Diffusion](https://github.com/Stability-AI/StableDiffusion) and [GLIGEN](https://github.com/gligen/GLIGEN) for their awesome models.


## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@inproceedings{ShilongLiu2023GroundingDM,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Shilong Liu and Zhaoyang Zeng and Tianhe Ren and Feng Li and Hao Zhang and Jie Yang and Chunyuan Li and Jianwei Yang and Hang Su and Jun Zhu and Lei Zhang},
  year={2023}
}
```




