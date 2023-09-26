# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import sys
import os
import time
sys.path.append(os.getcwd())

def test_extend_workflow():
    from lowCodeLLM import lowCodeLLM
    cases = json.load(open("./test/testcases/extend_workflow_test_cases.json", "r"))
    llm = lowCodeLLM(0.5, 0)
    for c in cases:
        task_prompt = c["task_prompt"]
        current_workflow = c["current_workflow"]
        step = c["step"]
        result = llm.extend_workflow(task_prompt, current_workflow, step)
        time.sleep(5)
        assert len(json.loads(result)) >= 1