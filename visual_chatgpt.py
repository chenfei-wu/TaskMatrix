import os
import gradio as gr
import random
import torch
import cv2
import re
import uuid
from PIL import Image
import numpy as np
import argparse

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

VISUAL_CHATGPT_PREFIX = """Visual ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Visual ChatGPT is able to process and understand large amounts of text and images. As a language model, Visual ChatGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual ChatGPT can invoke different tools to indirectly understand pictures. When talking about images, Visual ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual ChatGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to Visual ChatGPT with a description. The description helps Visual ChatGPT to understand this image, but Visual ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Visual ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

Visual ChatGPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Visual ChatGPT is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""

os.makedirs('image', exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png'
    return os.path.join(head, new_file_name)


class MaskFormer:
    def __init__(self, device):
        print(f"Initializing MaskFormer to {device}")
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    def inference(self, image_path, text):
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(original_image.size)


class ImageEditing:
    def __init__(self, device):
        print(f"Initializing ImageEditing to {device}")
        self.device = device
        self.mask_former = MaskFormer(device=self.device)
        self.revision = 'fp16' if 'cuda' in device else None
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype).to(device)

    @prompts(name="Remove Something From The Photo",
             description="useful when you want to remove and object or something from the photo "
                         "from its description or location. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the object need to be removed. ")
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        return self.inference_replace(f"{image_path},{to_be_removed_txt},background")

    @prompts(name="Replace Something From The Photo",
             description="useful when you want to replace an object from the object description or "
                         "location with another object from its description. "
                         "The input to this tool should be a comma separated string of three, "
                         "representing the image_path, the object to be replaced, the object to be replaced with ")
    def inference_replace(self, inputs):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")
        original_image = Image.open(image_path)
        original_size = original_image.size
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpaint(prompt=replace_with_txt, image=original_image.resize((512, 512)),
                                     mask_image=mask_image.resize((512, 512))).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="replace-something")
        updated_image = updated_image.resize(original_size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
            f"Output Image: {updated_image_path}")
        return updated_image_path


class InstructPix2Pix:
    def __init__(self, device):
        print(f"Initializing InstructPix2Pix to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                           safety_checker=None,
                                                                           torch_dtype=self.torch_dtype).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    @prompts(name="Instruct Image Using Text",
             description="useful when you want to the style of the image to be like the text. "
                         "like: make it look like a painting. or make it like a robot. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the text. ")
    def inference(self, inputs):
        """Change style of image."""
        print("===>Starting InstructPix2Pix Inference")
        image_path, text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        original_image = Image.open(image_path)
        image = self.pipe(text, image=original_image, num_inference_steps=40, image_guidance_scale=1.2).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        image.save(updated_image_path)
        print(f"\nProcessed InstructPix2Pix, Input Image: {image_path}, Instruct Text: {text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                            torch_dtype=self.torch_dtype)
        self.pipe.to(device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        prompt = text + ', ' + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename


class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype).to(self.device)

    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions


class Image2Canny:
    def __init__(self, device):
        print("Initializing Image2Canny")
        self.low_threshold = 100
        self.high_threshold = 200

    @prompts(name="Edge Detection On Image",
             description="useful when you want to detect the edge of the image. "
                         "like: detect the edges of this image, or canny detection on image, "
                         "or perform edge detection on this image, or detect the canny image of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        image = np.array(image)
        canny = cv2.Canny(image, self.low_threshold, self.high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        updated_image_path = get_new_image_name(inputs, func_name="edge")
        canny.save(updated_image_path)
        print(f"\nProcessed Image2Canny, Input Image: {inputs}, Output Text: {updated_image_path}")
        return updated_image_path


class CannyText2Image:
    def __init__(self, device):
        print(f"Initializing CannyText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-canny",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                            'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Canny Image",
             description="useful when you want to generate a new real image from both the user description and a canny image."
                         " like: generate a real image of a object or something from this canny image,"
                         " or generate a new real image of a object or something from this edge image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description. ")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="canny2image")
        image.save(updated_image_path)
        print(f"\nProcessed CannyText2Image, Input Canny: {image_path}, Input Text: {instruct_text}, "
              f"Output Text: {updated_image_path}")
        return updated_image_path


class Image2Line:
    def __init__(self, device):
        print("Initializing Image2Line")
        self.detector = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    @prompts(name="Line Detection On Image",
             description="useful when you want to detect the straight line of the image. "
                         "like: detect the straight lines of this image, or straight line detection on image, "
                         "or perform straight line detection on this image, or detect the straight line image of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        mlsd = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="line-of")
        mlsd.save(updated_image_path)
        print(f"\nProcessed Image2Line, Input Image: {inputs}, Output Line: {updated_image_path}")
        return updated_image_path


class LineText2Image:
    def __init__(self, device):
        print(f"Initializing LineText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-mlsd",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                            'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Line Image",
             description="useful when you want to generate a new real image from both the user description "
                         "and a straight line image. "
                         "like: generate a real image of a object or something from this straight line image, "
                         "or generate a new real image of a object or something from this straight lines. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description. ")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="line2image")
        image.save(updated_image_path)
        print(f"\nProcessed LineText2Image, Input Line: {image_path}, Input Text: {instruct_text}, "
              f"Output Text: {updated_image_path}")
        return updated_image_path


class Image2Hed:
    def __init__(self, device):
        print("Initializing Image2Hed")
        self.detector = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    @prompts(name="Hed Detection On Image",
             description="useful when you want to detect the soft hed boundary of the image. "
                         "like: detect the soft hed boundary of this image, or hed boundary detection on image, "
                         "or perform hed boundary detection on this image, or detect soft hed boundary image of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        hed = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="hed-boundary")
        hed.save(updated_image_path)
        print(f"\nProcessed Image2Hed, Input Image: {inputs}, Output Hed: {updated_image_path}")
        return updated_image_path


class HedText2Image:
    def __init__(self, device):
        print(f"Initializing HedText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-hed",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                            'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Soft Hed Boundary Image",
             description="useful when you want to generate a new real image from both the user description "
                         "and a soft hed boundary image. "
                         "like: generate a real image of a object or something from this soft hed boundary image, "
                         "or generate a new real image of a object or something from this hed boundary. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="hed2image")
        image.save(updated_image_path)
        print(f"\nProcessed HedText2Image, Input Hed: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Image2Scribble:
    def __init__(self, device):
        print("Initializing Image2Scribble")
        self.detector = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    @prompts(name="Sketch Detection On Image",
             description="useful when you want to generate a scribble of the image. "
                         "like: generate a scribble of this image, or generate a sketch from this image, "
                         "detect the sketch from this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        scribble = self.detector(image, scribble=True)
        updated_image_path = get_new_image_name(inputs, func_name="scribble")
        scribble.save(updated_image_path)
        print(f"\nProcessed Image2Scribble, Input Image: {inputs}, Output Scribble: {updated_image_path}")
        return updated_image_path


class ScribbleText2Image:
    def __init__(self, device):
        print(f"Initializing ScribbleText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-scribble",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                            'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Sketch Image",
             description="useful when you want to generate a new real image from both the user description and "
                         "a scribble image or a sketch image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="scribble2image")
        image.save(updated_image_path)
        print(f"\nProcessed ScribbleText2Image, Input Scribble: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Image2Pose:
    def __init__(self, device):
        print("Initializing Image2Pose")
        self.detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    @prompts(name="Pose Detection On Image",
             description="useful when you want to detect the human pose of the image. "
                         "like: generate human poses of this image, or generate a pose image from this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        pose = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="human-pose")
        pose.save(updated_image_path)
        print(f"\nProcessed Image2Pose, Input Image: {inputs}, Output Pose: {updated_image_path}")
        return updated_image_path


class PoseText2Image:
    def __init__(self, device):
        print(f"Initializing PoseText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.num_inference_steps = 20
        self.seed = -1
        self.unconditional_guidance_scale = 9.0
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Pose Image",
             description="useful when you want to generate a new real image from both the user description "
                         "and a human pose image. "
                         "like: generate a real image of a human from this human pose image, "
                         "or generate a new real image of a human from this pose. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pose2image")
        image.save(updated_image_path)
        print(f"\nProcessed PoseText2Image, Input Pose: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Image2Seg:
    def __init__(self, device):
        print("Initializing Image2Seg")
        self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        self.ade_palette = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                            [102, 255, 0], [92, 0, 255]]

    @prompts(name="Segmentation On Image",
             description="useful when you want to detect segmentations of the image. "
                         "like: segment this image, or generate segmentations on this image, "
                         "or perform segmentation on this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        palette = np.array(self.ade_palette)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        segmentation = Image.fromarray(color_seg)
        updated_image_path = get_new_image_name(inputs, func_name="segmentation")
        segmentation.save(updated_image_path)
        print(f"\nProcessed Image2Pose, Input Image: {inputs}, Output Pose: {updated_image_path}")
        return updated_image_path


class SegText2Image:
    def __init__(self, device):
        print(f"Initializing SegText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-seg",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Segmentations",
             description="useful when you want to generate a new real image from both the user description and segmentations. "
                         "like: generate a real image of a object or something from this segmentation image, "
                         "or generate a new real image of a object or something from these segmentations. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        image.save(updated_image_path)
        print(f"\nProcessed SegText2Image, Input Seg: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Image2Depth:
    def __init__(self, device):
        print("Initializing Image2Depth")
        self.depth_estimator = pipeline('depth-estimation')

    @prompts(name="Predict Depth On Image",
             description="useful when you want to detect depth of the image. like: generate the depth from this image, "
                         "or detect the depth map on this image, or predict the depth for this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        depth = self.depth_estimator(image)['depth']
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)
        depth = Image.fromarray(depth)
        updated_image_path = get_new_image_name(inputs, func_name="depth")
        depth.save(updated_image_path)
        print(f"\nProcessed Image2Depth, Input Image: {inputs}, Output Depth: {updated_image_path}")
        return updated_image_path


class DepthText2Image:
    def __init__(self, device):
        print(f"Initializing DepthText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-depth", torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Depth",
             description="useful when you want to generate a new real image from both the user description and depth image. "
                         "like: generate a real image of a object or something from this depth image, "
                         "or generate a new real image of a object or something from the depth map. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="depth2image")
        image.save(updated_image_path)
        print(f"\nProcessed DepthText2Image, Input Depth: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Image2Normal:
    def __init__(self, device):
        print("Initializing Image2Normal")
        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        self.bg_threhold = 0.4

    @prompts(name="Predict Normal Map On Image",
             description="useful when you want to detect norm map of the image. "
                         "like: generate normal map from this image, or predict normal map of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        original_size = image.size
        image = self.depth_estimator(image)['predicted_depth'][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < self.bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < self.bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize(original_size)
        updated_image_path = get_new_image_name(inputs, func_name="normal-map")
        image.save(updated_image_path)
        print(f"\nProcessed Image2Normal, Input Image: {inputs}, Output Depth: {updated_image_path}")
        return updated_image_path


class NormalText2Image:
    def __init__(self, device):
        print(f"Initializing NormalText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Normal Map",
             description="useful when you want to generate a new real image from both the user description and normal map. "
                         "like: generate a real image of a object or something from this normal map, "
                         "or generate a new real image of a object or something from the normal map. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="normal2image")
        image.save(updated_image_path)
        print(f"\nProcessed NormalText2Image, Input Normal: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class VisualQuestionAnswering:
    def __init__(self, device):
        print(f"Initializing VisualQuestionAnswering to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype).to(self.device)

    @prompts(name="Answer Question About The Image",
             description="useful when you need an answer for a question based on an image. "
                         "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the question")
    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
              f"Output Answer: {answer}")
        return answer


class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")

        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

        self.models = {}
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': VISUAL_CHATGPT_PREFIX, 'format_instructions': VISUAL_CHATGPT_FORMAT_INSTRUCTIONS,
                          'suffix': VISUAL_CHATGPT_SUFFIX}, )

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state

    def run_image(self, image, state, txt):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models['ImageCaptioning'].inference(image_filename)
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, f'{txt} {image_filename} '


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default="ImageCaptioning_cuda:0,Text2Image_cuda:0")
    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    bot = ConversationBot(load_dict=load_dict)
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="Visual ChatGPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["image"])

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        demo.launch(server_name="0.0.0.0", server_port=7868)
