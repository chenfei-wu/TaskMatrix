# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from flask import Flask, request, send_from_directory
from flask_cors import CORS, cross_origin
from lowCodeLLM import lowCodeLLM
from flask.logging import default_handler
import logging

app = Flask('lowcode-llm', static_folder='', template_folder='')
app.debug = True
llm = lowCodeLLM()
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger = gunicorn_logger
logging_format = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
default_handler.setFormatter(logging_format)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route('/api/get_workflow', methods=['POST'])
@cross_origin()
def get_workflow():
    try:
        request_content = request.get_json()
        task_prompt = request_content['task_prompt']
        workflow = llm.get_workflow(task_prompt)
        return workflow, 200
    except Exception as e:
        app.logger.error(
            'failed to get_workflow, msg:%s, request data:%s' % (str(e), request.json))
        return {'errmsg': str(e)}, 500

@app.route('/api/extend_workflow', methods=['POST'])
@cross_origin()
def extend_workflow():
    try:
        request_content = request.get_json()
        task_prompt = request_content['task_prompt']
        current_workflow = request_content['current_workflow']
        step = request_content['step']
        sub_workflow = llm.extend_workflow(task_prompt, current_workflow, step)
        return sub_workflow, 200
    except Exception as e:
        app.logger.error(
            'failed to extend_workflow, msg:%s, request data:%s' % (str(e), request.json))
        return {'errmsg': str(e)}, 500

@app.route('/api/execute', methods=['POST'])
@cross_origin()
def execute():
    try:
        request_content = request.get_json()
        task_prompt = request_content['task_prompt']
        confirmed_workflow = request_content['confirmed_workflow']
        curr_input = request_content['curr_input']
        history = request_content['history']
        response = llm.execute(task_prompt,confirmed_workflow, history, curr_input)
        return response, 200
    except Exception as e:
        app.logger.error(
            'failed to execute, msg:%s, request data:%s' % (str(e), request.json))
        return {'errmsg': str(e)}, 500