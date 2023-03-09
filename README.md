# Visual ChatGPT 

**Visual ChatGPT** connects ChatGPT and a series of Visual Foundation Models to enable **sending** and **receiving** images during chatting.

See our paper: [<font size=5>Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models</font>](https://arxiv.org/abs/2303.04671)

## Demo 
<img src="./assets/demo_short.gif" width="750">

##  System Architecture 

 
<p align="center"><img src="./assets/figure.jpg" alt="Logo"></p>


## Quick Start

```
# create a new environment
conda create -n visgpt python=3.8

#  prepare the basic environments
pip install -r requirement.txt

# download the visual foundation models
bash download.sh

# prepare your private openAI private key
export OPENAI_API_KEY={Your_Private_Openai_Key}

# create a folder to save images
mkdir ./image

# Start Visual ChatGPT !
python visual_chatgpt.py
```


## Acknowledgement
We appreciate the open source of the following projects:

- HuggingFace [[Project]](https://github.com/huggingface/transformers)

- ControlNet  [[Paper]](https://arxiv.org/abs/2302.05543) [[Project]](https://github.com/lllyasviel/ControlNet)

- Stable Diffusion [[Paper]](https://arxiv.org/abs/2112.10752)  [[Project]](https://github.com/CompVis/stable-diffusion)
