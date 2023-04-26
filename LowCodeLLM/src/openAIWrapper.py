import os
import openai

class OpenAIWrapper:
    def __init__(self, temperature):
        self.key = os.environ.get("OPENAIKEY")
        openai.api_key = self.key

        # Access the USE_AZURE environment variable
        self.use_azure = os.environ.get('USE_AZURE')

        # Check if USE_AZURE is defined
        if self.use_azure is not None:
            # Convert the USE_AZURE value to boolean
            self.use_azure = self.use_azure.lower() == 'true'
        else:
            self.use_azure = False

        if self.use_azure:
            openai.api_type = "azure"
            self.api_base = os.environ.get('API_BASE')
            openai.api_base = self.api_base
            self.api_version = os.environ.get('API_VERSION')
            openai.api_version = self.api_version
            self.engine = os.environ.get('MODEL')
        else:
            self.chat_model_id = "gpt-3.5-turbo"
            
        self.temperature = temperature
        self.max_tokens = 2048
        self.top_p = 1
        self.time_out = 7
    
    def run(self, prompt):
        return self._post_request_chat(prompt)

    def _post_request_chat(self, messages):
        try:
            if self.use_azure:
                response = openai.ChatCompletion.create(
                    engine=self.engine,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            else:
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
