# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from openAIWrapper import OpenAIWrapper

EXECUTING_LLM_PREFIX = """Executing LLM is designed to provide outstanding responses.
Executing LLM will be given a overall task as the background of the conversation between the Executing LLM and human.
When providing response, Executing LLM MUST STICTLY follow the provided standard operating procedure (SOP).
the SOP is formatted as:
'''
STEP 1: [step name][step descriptions][[[if 'condition1'][Jump to STEP]], [[if 'condition2'][Jump to STEP]], ...]
'''
here "[[[if 'condition1'][Jump to STEP n]]]" is judgmental logic. It means when you're performing this step, and if 'condition1' is satisfied, you will perform STEP n next.

Remember: 
Executing LLM is facing a real human, who does not know what SOP is. 
So, Do not show him/her the SOP steps you are following, or it will make him/her confused. Just response the answer.
"""

EXECUTING_LLM_SUFFIX = """
Remember: 
Executing LLM is facing a real human, who does not know what SOP is. 
So, Do not show him/her the SOP steps you are following, or it will make him/her confused. Just response the answer.
"""

class executingLLM:
    def __init__(self, temperature) -> None:
        self.prefix = EXECUTING_LLM_PREFIX
        self.suffix = EXECUTING_LLM_SUFFIX
        self.LLM = OpenAIWrapper(temperature)
        self.messages = [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "system", "content": self.prefix}]

    def execute(self, current_prompt, history):
        ''' provide LLM the dialogue history and the current prompt to get response '''
        messages = self.messages + history
        messages.append({'role': 'user', "content": current_prompt + self.suffix})
        response, status = self.LLM.run(messages)
        if status:
            return response
        else:
            return "OpenAI API error."