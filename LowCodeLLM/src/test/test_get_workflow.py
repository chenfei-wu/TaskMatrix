# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import sys
import os
sys.path.append(os.getcwd())

def test_get_workflow():
    from lowCodeLLM import lowCodeLLM
    cases = json.load(open("./test/testcases/get_workflow_test_cases.json", "r"))
    llm = lowCodeLLM(0.5, 0)
    for c in cases:
        task_prompt = c["task_prompt"]
        result = llm.get_workflow(task_prompt)
        assert len(json.loads(result)) >= 1