# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import sys
import os
import time
sys.path.append(os.getcwd())

def test_extend_workflow():
    from lowCodeLLM import lowCodeLLM
    cases = json.load(open("./test/testcases/execute_test_cases.json", "r"))
    llm = lowCodeLLM(0.5, 0)
    for c in cases:
        task_prompt = c["task_prompt"]
        confirmed_workflow = c["confirmed_workflow"]
        history = c["history"]
        curr_input = c["curr_input"]
        result = llm.execute(task_prompt, confirmed_workflow, history, curr_input)
        time.sleep(5)
        assert type(result) == str
        assert len(result) > 0