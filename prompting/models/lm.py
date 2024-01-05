import os
import sys
import json

import openai

PROJECT_ROOT = os.path.join(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import defs
import prompt
import dsl.parser as parser

class LanguageModel:
    def __init__(self, model="gpt-4"):
        secrets_file = os.path.join(PROJECT_ROOT, "secrets.json")
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
        # request = {
        #     "messages": messages,
        #     "model": self.model,
        #     "n": n_responses,
        # }
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            n=n_responses,
        )
        # print(response)
        # Save the response to a file
        if log:
            timestamp = defs.get_timestamp(micros=False)
            log_dir = os.path.join(PROJECT_ROOT, "models/logs")
            dump = response.model_dump_json(indent=4)
            with open(os.path.join(log_dir, f"{timestamp}.json"), "w") as f:
                f.write(dump)

        finish_error = 0
        for choice in response.choices:
            if choice.finish_reason != "stop":
                finish_error += 1
        if finish_error > 0:
            print(f"WARNING: {finish_error} responses did not finish by 'stop'")

        completions = [choice.message.content for choice in response.choices]
        return completions
    
def query_task(task_id, n_responses, output_dir, model="gpt-4"):
    """
    Returns a list of n_responses responses to the prompt
    """
    pmpt = prompt.build_prompt(task_id)
    lm = LanguageModel(model=model)
    system_prompt_path = os.path.join(PROJECT_ROOT, "models/templates/system.txt")
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    completions = lm.query(pmpt, n_responses, system_prompt=system_prompt, log=True)
    correct, incorrect = [], []
    # split the completions
    for completion in completions:
        full_completion = completion
        # split by the code block delimiter
        completion = completion.split("```")
        # trim and remove empty strings
        completion = [x.strip() for x in completion if x.strip() != ""]
        if len(completion) == 0:
            print("Error: empty completion")
            continue
        # keep only syntactically correct completions
        correct_completions = []
        for c in completion:
            try:
                parser.Parser(None).parse_tree(c)
                correct_completions.append(c)
            except Exception as e:
                pass
        if len(correct_completions) == 0:
            incorrect.append(full_completion)
        else:  
            # non-deterministic: keep only the last syntactically correct completion
            correct.append(correct_completions[-1])
    # print the number of correct and incorrect completions
    print(f"Task {task_id}")
    print(f"Correct: {len(correct)}")
    print(f"Incorrect: {len(incorrect)}")
    # save the completions to files
    timestamp = defs.get_timestamp(micros=False)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"{task_id}_correct.txt"), "w") as f:
        f.write("\n\n".join(correct))
    with open(os.path.join(output_dir, f"{task_id}_incorrect.txt"), "w") as f:
        f.write("\n\n".join(incorrect))
    return completions

def query_task_list(task_ids, n_responses):
    # create a directory for the output
    timestamp = defs.get_timestamp(micros=False)
    output_dir = os.path.join(PROJECT_ROOT, "models/logs", f"gens_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # query each task
    for task_id in task_ids:
        query_task(task_id, n_responses, output_dir)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python lm.py <task_list_file>")
        exit(1)
    task_list_file = args[0]
    with open(task_list_file, "r") as f:
        task_ids = f.read().splitlines()
    query_task_list(task_ids, 30)