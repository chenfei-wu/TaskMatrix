import os
import openai

class OpenAIWrapper:
    def __init__(self, temperature):
        self.key = os.environ.get("OPENAIKEY")
        self.chat_model_id = "gpt-3.5-turbo"
        self.temperature = temperature
        self.max_tokens = 2048
        self.top_p = 1
        self.time_out = 7
    
    def run(self, prompt):
        return self._post_request_chat(prompt)

    def _post_request_chat(self, messages):
        try:
            openai.api_key = self.key
            response = openai.ChatCompletion.create(
                model=self.chat_model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                frequency_penalty=0,
                presence_penalty=0
            )
            res = response['choices'][0]['message']['content']
            return res, True
        except Exception as e:
            return "", False
