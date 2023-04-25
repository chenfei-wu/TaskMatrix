# Low-code LLM

**Low-code LLM** is a novel human-LLM interaction pattern, involving human in the loop to achieve more controllable and stable responses.

See our paper: [Low-code LLM: Visual Programming over LLMs](https://arxiv.org/abs/2304.08103)

In the future, [TaskMatrix.AI](https://arxiv.org/abs/2304.08103) can enhance task automation by breaking down tasks more effectively and utilizing existing foundation models and APIs of other AI models and systems to achieve diversified tasks in both digital and physical domains. And the low-code human-LLM interaction pattern can enhance user's experience on controling over the process and expressing their preference.

## Video Demo
https://user-images.githubusercontent.com/43716920/233937121-cd057f04-dec8-45b8-9c52-a9e9594eec80.mp4

(This is a conceptual video demo to demonstrate the complete process)

## Quick Start
Please note that due to time constraints, the code we provide is only the minimum viable version of the low-code LLM interactive code, i.e. only demonstrating the core concept of Low-code LLM human-LLM interaction. We welcome anyone who is interested in improving our front-end interface.
Currently, both the `OpenAI API` and `Azure OpenAI Service` are supported. You would be required to provide the requisite information to invoke these APIs.

```
# clone the repo
git clone https://github.com/microsoft/TaskMatrix.git

# go to directlory
cd LowCodeLLM

# build and run docker
docker build -t lowcode:latest .

# If OpenAI API is being used, it is only necessary to provide the API key.
docker run -p 8888:8888 --env OPENAIKEY={Your_Private_Openai_Key} lowcode:latest

# When using Azure OpenAI Service, it is advisable to store the necessary information in a configuration file for ease of access.
# Kindly duplicate the config.template file and name the copied file as config.ini. Then, fill out the necessary information in the config.ini file.
docker run -p 8888:8888 --env-file config.ini lowcode:latest
```
You can now try it by visiting [Demo page](http://localhost:8888/)


## System Overview

<img src="https://github.com/microsoft/TaskMatrix/blob/main/assets/low-code-llm.png" alt="overview" width="800"/>

As shown in the above figure, human-LLM interaction can be completed by:
- A Planning LLM that generates a highly structured workflow for complex tasks.
- Users editing the workflow with predefined low-code operations, which are all supported by clicking, dragging, or text editing. 
- An Executing LLM that generates responses with the reviewed workflow. 
- Users continuing to refine the workflow until satisfactory results are obtained.

## Six Kinds of Pre-defined Low-code Operations
<img src="https://github.com/microsoft/TaskMatrix/blob/main/assets/low-code-operation.png" alt="operations" width="800"/>

## Advantages

- **Controllable Generation.** Complicated tasks are decomposed into structured conducting plans and presented to users as workflows. Users can control the LLMs’ execution through low-code operations to achieve more controllable responses. The responses generated followed the customized workflow will be more aligned with the user’s requirements.
- **Friendly Interaction.** The intuitive workflow enables users to swiftly comprehend the LLMs' execution logic, and the low-code operation through a graphical user interface empowers users to conveniently modify the workflow in a user-friendly manner. In this way, time-consuming prompt engineering is mitigated, allowing users to efficiently implement their ideas into detailed instructions to achieve high-quality results.
- **Wide applicability.** The proposed framework can be applied to a wide range of complex tasks across various domains, especially in situations where human's intelligence or preference are indispensable.


## Acknowledgement
Part of this paper has been collaboratively crafted through interactions with the proposed Low-code LLM. The process began with GPT-4 outlining the framework, followed by the authors supplementing it with innovative ideas and refining the structure of the workflow. Ultimately, GPT-4 took charge of generating cohesive and compelling text.
