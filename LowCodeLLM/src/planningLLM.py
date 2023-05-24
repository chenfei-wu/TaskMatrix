# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import json
from openAIWrapper import OpenAIWrapper

PLANNING_LLM_PREFIX = """Planning LLM is designed to provide a standard operating procedure so that an difficult task will be broken down into several steps, and the task will be easily solved by following these steps.
Planning LLM is a powerful problem-solving assistant, and it only needs to analyze the task and provide standard operating procedure as guidance, but does not need actually to solve the problem.
Sometimes there exists some unknown or undetermined situation, thus judgmental logic is needed: some "conditions" are listed, and the next step that should be carried out if a "condition" is satisfied is also listed. The judgmental logics are not necessary.
Planning LLM MUST only provide standard operating procedure in the following format without any other words:
'''
STEP 1: [step name][step descriptions][[[if 'condition1'][Jump to STEP]], [[[if 'condition1'][Jump to STEP]], [[if 'condition2'][Jump to STEP]], ...]
STEP 2: [step name][step descriptions][[[if 'condition1'][Jump to STEP]], [[[if 'condition1'][Jump to STEP]], [[if 'condition2'][Jump to STEP]], ...]
...
'''

For example:
'''
STEP 1: [Brainstorming][Choose a topic or prompt, and generate ideas and organize them into an outline][]
STEP 2: [Research][Gather information, take notes and organize them into the outline][[[lack of ideas][Jump to STEP 1]]]
...
'''
"""

EXTEND_PREFIX = """
\nsome steps of the SOP provided by Planning LLM are too rough, so Planning LLM can also provide a detailed sub-SOP for the given step.
Remember, Planning LLM take the overall SOP into consideration, and the sub-SOP MUST be consistent with the rest of the steps, and there MUST be no duplication in content between the extension and the original SOP.

For example:
If the overall SOP is:
'''
STEP 1: [Brainstorming][Choose a topic or prompt, and generate ideas and organize them into an outline][]
STEP 2: [Write][write the text][]
'''
If the STEP 2: "write the text" is too rough and needs to be extended, then the response could be:
'''
STEP 2.1: [Write the title][write the title of the essay][]
STEP 2.2: [Write the body][write the body of the essay][[[if lack of materials][Jump to STEP 1]]]
'''
"""

PLANNING_LLM_SUFFIX = """\nRemember: Planning LLM is very strict to the format and NEVER reply any word other than the standard operating procedure.
"""

class planningLLM:
    def __init__(self, temperature) -> None:
        self.prefix = PLANNING_LLM_PREFIX
        self.suffix = PLANNING_LLM_SUFFIX
        self.LLM = OpenAIWrapper(temperature)
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def get_workflow(self, task_prompt):
        '''
        - input: task_prompt
        - output: workflow (json)
        '''
        messages = self.messages + [{'role': 'user', "content": PLANNING_LLM_PREFIX+'\nThe task is:\n'+task_prompt+PLANNING_LLM_SUFFIX}]
        response, status = self.LLM.run(messages)
        if status:
            return self._txt2json(response)
        else:
            return "OpenAI API error."

    def extend_workflow(self, task_prompt, current_workflow, step):
        messages = self.messages + [{'role': 'user', "content": PLANNING_LLM_PREFIX+'\nThe task is:\n'+task_prompt+PLANNING_LLM_SUFFIX}]
        messages.append({'role': 'user', "content": EXTEND_PREFIX+
                         'The current SOP is:\n'+current_workflow+
                         '\nThe step needs to be extended is:\n'+step+
                         PLANNING_LLM_SUFFIX})
        response, status = self.LLM.run(messages)
        if status:
            return self._txt2json(response)
        else:
            return "OpenAI API error."

    def _txt2json(self, workflow_txt):
        ''' convert the workflow in natural language to json format '''
        workflow = []
        try:
            steps = workflow_txt.split('\n')
            for step in steps:
                if step[0:4] != "STEP":
                    continue
                left_indices = [_.start() for _ in re.finditer("\[", step)]
                right_indices = [_.start() for _ in re.finditer("\]", step)]
                step_id = step[: left_indices[0]-2]
                step_name = step[left_indices[0]+1: right_indices[0]]
                step_description = step[left_indices[1]+1: right_indices[1]]
                jump_str = step[left_indices[2]+1: right_indices[-1]]
                if re.findall(re.compile(r'[A-Za-z]',re.S), jump_str) == []:
                    workflow.append({"stepId": step_id, "stepName": step_name, "stepDescription": step_description, "jumpLogic": [], "extension": []})
                    continue
                jump_logic = []
                left_indices = [_.start() for _ in re.finditer('\[', jump_str)]
                right_indices = [_.start() for _ in re.finditer('\]', jump_str)]
                i = 1
                while i < len(left_indices):
                    jump = {"Condition": jump_str[left_indices[i]+1: right_indices[i-1]], "Target": re.search(r'STEP\s\d', jump_str[left_indices[i+1]+1: right_indices[i]]).group(0)}
                    jump_logic.append(jump)
                    i += 3
                workflow.append({"stepId": step_id, "stepName": step_name, "stepDescription": step_description, "jumpLogic": jump_logic, "extension": []})
            return json.dumps(workflow)
        except:
            print("Format error, please try again.")