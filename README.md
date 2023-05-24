# TaskMatrix

**TaskMatrix** connects ChatGPT and a series of Visual Foundation Models to enable **sending** and **receiving** images during chatting.

See our paper: [<font size=5>Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models</font>](https://arxiv.org/abs/2303.04671)

<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" href="https://huggingface.co/spaces/microsoft/visual_chatgpt">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" alt="Open in Spaces">
</a>

<a src="https://colab.research.google.com/assets/colab-badge.svg" href="https://colab.research.google.com/drive/1P3jJqKEWEaeNcZg8fODbbWeQ3gxOHk2-?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

## Updates:
- Now TaskMatrix supports [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [segment-anything](https://github.com/facebookresearch/segment-anything)! Thanks **@jordddan** for his efforts. For the image editing case, `GroundingDINO` is first used to locate bounding boxes guided by given text, then `segment-anything` is used to generate the related mask, and finally stable diffusion inpainting is used to edit image based on the mask. 
    - Firstly, run `python visual_chatgpt.py --load "Text2Box_cuda:0,Segmenting_cuda:0,Inpainting_cuda:0,ImageCaptioning_cuda:0"`
    - Then, say `find xxx in the image` or `segment xxx in the image`. `xxx` is an object. TaskMatrix will return the detection or segmentation result!


- Now TaskMatrix can support Chinese! Thanks to **@Wang-Xiaodong1899** for his efforts.
- We propose the **template** idea in TaskMatrix!
    - A template is a **pre-defined execution flow** that assists ChatGPT in assembling complex tasks involving multiple foundation models. 
    - A template contains the **experiential solution** to complex tasks as determined by humans. 
    - A template can **invoke multiple foundation models** or even **establish a new ChatGPT session**
    - To define a **template**, simply adding a class with attributes `template_model = True`
- Thanks to **@ShengmingYin** and **@thebestannie** for providing a template example in `InfinityOutPainting` class (see the following gif)
    - Firstly, run `python visual_chatgpt.py --load "Inpainting_cuda:0,ImageCaptioning_cuda:0,VisualQuestionAnswering_cuda:0"`
    - Secondly, say `extend the image to 2048x1024` to TaskMatrix!
    - By simply creating an `InfinityOutPainting` template, TaskMatrix can seamlessly extend images to any size through collaboration with existing `ImageCaptioning`, `Inpainting`, and `VisualQuestionAnswering` foundation models, **without the need for additional training**.
- **TaskMatrix needs the effort of the community! We crave your contribution to add new and interesting features!**
<img src="./assets/demo_inf.gif" width="750">


## Insight & Goal:
On the one hand, **ChatGPT (or LLMs)** serves as a **general interface** that provides a broad and diverse understanding of a
wide range of topics. On the other hand, **Foundation Models** serve as **domain experts** by providing deep knowledge in specific domains.
By leveraging **both general and deep knowledge**, we aim at building an AI that is capable of handling various tasks.


## Demo 
<img src="./assets/demo_short.gif" width="750">

##  System Architecture 

 
<p align="center"><img src="./assets/figure.jpg" alt="Logo"></p>


## Quick Start

```
# clone the repo
git clone https://github.com/microsoft/TaskMatrix.git

# Go to directory
cd visual-chatgpt

# create a new environment
conda create -n visgpt python=3.8

# activate the new environment
conda activate visgpt

#  prepare the basic environments
pip install -r requirements.txt
pip install  git+https://github.com/IDEA-Research/GroundingDINO.git
pip install  git+https://github.com/facebookresearch/segment-anything.git

# prepare your private OpenAI key (for Linux)
export OPENAI_API_KEY={Your_Private_Openai_Key}

# prepare your private OpenAI key (for Windows)
set OPENAI_API_KEY={Your_Private_Openai_Key}

# Start TaskMatrix !
# You can specify the GPU/CPU assignment by "--load", the parameter indicates which 
# Visual Foundation Model to use and where it will be loaded to
# The model and device are separated by underline '_', the different models are separated by comma ','
# The available Visual Foundation Models can be found in the following table
# For example, if you want to load ImageCaptioning to cpu and Text2Image to cuda:0
# You can use: "ImageCaptioning_cpu,Text2Image_cuda:0"

# Advice for CPU Users
python visual_chatgpt.py --load ImageCaptioning_cpu,Text2Image_cpu

# Advice for 1 Tesla T4 15GB  (Google Colab)                       
python visual_chatgpt.py --load "ImageCaptioning_cuda:0,Text2Image_cuda:0"
                                
# Advice for 4 Tesla V100 32GB                            
python visual_chatgpt.py --load "Text2Box_cuda:0,Segmenting_cuda:0,
    Inpainting_cuda:0,ImageCaptioning_cuda:0,
    Text2Image_cuda:1,Image2Canny_cpu,CannyText2Image_cuda:1,
    Image2Depth_cpu,DepthText2Image_cuda:1,VisualQuestionAnswering_cuda:2,
    InstructPix2Pix_cuda:2,Image2Scribble_cpu,ScribbleText2Image_cuda:2,
    SegText2Image_cuda:2,Image2Pose_cpu,PoseText2Image_cuda:2,
    Image2Hed_cpu,HedText2Image_cuda:3,Image2Normal_cpu,
    NormalText2Image_cuda:3,Image2Line_cpu,LineText2Image_cuda:3"

```

## GPU memory usage
Here we list the GPU memory usage of each visual foundation model, you can specify which one you like:

| Foundation Model        | GPU Memory (MB) |
|------------------------|-----------------|
| ImageEditing           | 3981            |
| InstructPix2Pix        | 2827            |
| Text2Image             | 3385            |
| ImageCaptioning        | 1209            |
| Image2Canny            | 0               |
| CannyText2Image        | 3531            |
| Image2Line             | 0               |
| LineText2Image         | 3529            |
| Image2Hed              | 0               |
| HedText2Image          | 3529            |
| Image2Scribble         | 0               |
| ScribbleText2Image     | 3531            |
| Image2Pose             | 0               |
| PoseText2Image         | 3529            |
| Image2Seg              | 919             |
| SegText2Image          | 3529            |
| Image2Depth            | 0               |
| DepthText2Image        | 3531            |
| Image2Normal           | 0               |
| NormalText2Image       | 3529            |
| VisualQuestionAnswering| 1495            |

## Acknowledgement
We appreciate the open source of the following projects:

[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194; 
[ControlNet](https://github.com/lllyasviel/ControlNet) &#8194; 
[InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) &#8194; 
[CLIPSeg](https://github.com/timojl/clipseg) &#8194;
[BLIP](https://github.com/salesforce/BLIP) &#8194;

## Contact Information
For help or issues using the TaskMatrix, please submit a GitHub issue.

For other communications, please contact Chenfei WU (chewu@microsoft.com) or Nan DUAN (nanduan@microsoft.com).

## Trademark Notice

Trademarks This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.