import os
import json
import time

import openai

import defs

class LanguageModel:
    def __init__(self, model):
        secrets_file = os.path.join(defs.PROJECT_ROOT, "secrets.json")
        with open(secrets_file, "r") as f:
            secrets = json.load(f)
        openai.api_key = secrets["API_KEY"]
        self.client = openai.Client(api_key=secrets["API_KEY"])
        models_list = self.client.models.list()
        is_available = False
        for m in models_list:
            if m.id == model:
                is_available = True 
                break
        if not is_available:
            raise Exception(f"Model {model} is not available")
        # print(f"Using model {model}")
        self.model = model
    
    def query(self, prompt, n_responses, system_prompt="", log=True):
        """
        Returns a list of n_responses responses to the prompt
        """
        messages = [
            # system prompt
            {
                "role": "system",
                "content": system_prompt
            },
            # user prompt
            {
                "role": "user",
                "content": prompt
            }
        ]

        request_start = time.time()
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            n=n_responses,
            max_tokens=4000,
            # # this might impact request time significantly
            # logprobs=True,
            response_format={ "type": "json_object" }
        )
        request_end = time.time()
        request_time = request_end - request_start
        print(f"Request took {request_time:.2f} seconds")
        # Save the response to a file
        response = response.model_dump()
        response["request_time"] = request_time
        response["n_requested"] = n_responses
        response["prompt_messages"] = messages
        if log:
            timestamp = defs.get_timestamp(micros=False)
            log_dir = os.path.join(defs.PROJECT_ROOT, "models/logs/requests")
            # dump = response.model_dump_json(indent=4)
            dump = json.dumps(response, indent=4)
            with open(os.path.join(log_dir, f"{timestamp}.json"), "w") as f:
                f.write(dump)

        finish_error = 0
        for choice in response["choices"]:
            if choice["finish_reason"] != "stop":
                finish_error += 1
        if finish_error > 0:
            print(f"WARNING: {finish_error} responses did not finish by 'stop'")

        # completions = [choice.message.content for choice in response.choices]
        # return completions

        return response
    