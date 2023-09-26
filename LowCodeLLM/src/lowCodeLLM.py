# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from planningLLM import planningLLM
from executingLLM import executingLLM
import json

class lowCodeLLM:
    def __init__(self, PLLM_temperature=0.4, ELLM_temperature=0):
        self.PLLM = planningLLM(PLLM_temperature)
        self.ELLM = executingLLM(ELLM_temperature)

    def get_workflow(self, task_prompt):
        return self.PLLM.get_workflow(task_prompt)

    def extend_workflow(self, task_prompt, current_workflow, step=''):
        ''' generate a sub-workflow for one of steps 
            - input: the current workflow, the step needs to extend
            - output: sub-workflow '''
        workflow = self._json2txt(current_workflow)
        return self.PLLM.extend_workflow(task_prompt, workflow, step)

    def execute(self, task_prompt,confirmed_workflow, history, curr_input):
        ''' chat with the workflow-equipped low-code LLM '''
        prompt = [{'role': 'system', "content": 'The overall task you are facing is: '+task_prompt+
                '\nThe standard operating procedure(SOP) is:\n'+self._json2txt(confirmed_workflow)}]
        history = prompt + history
        response = self.ELLM.execute(curr_input, history)
        return response
    
    def _json2txt(self, workflow_json):
        ''' convert the json workflow to text'''
        def json2text_step(step):
            step_res = ""
            step_res += step["stepId"] + ": [" + step["stepName"] + "]"
            step_res += "[" + step["stepDescription"] + "]["
            for jump in step["jumpLogic"]:
                step_res += "[[" + jump["Condition"] + "][" + jump["Target"] + "]],"
            step_res += "]\n"
            return step_res

        workflow_txt = ""
        for step in json.loads(workflow_json):
            workflow_txt += json2text_step(step)
            for substep in step['extension']:
                workflow_txt += json2text_step(substep)
        return workflow_txt