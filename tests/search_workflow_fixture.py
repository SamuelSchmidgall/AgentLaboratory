"""Load real workflow/agent code without unrelated GPU, server or model SDKs.

Only external imports and the arXiv/model boundaries are substituted. Production
configuration parsing, construction, prompts, dispatch and history all execute.
This is a component test harness, not an end-to-end model or application run.
"""

import importlib
import json
import os
import re
import sys
import time
import types
from unittest.mock import Mock, patch


def module(name, **exports):
    result = types.ModuleType(name)
    result.__dict__.update(exports)
    return result


def load_workflow():
    dependencies = {
        "tiktoken": module("tiktoken"),
        "openai": module("openai", OpenAI=Mock()),
        "google": module("google"),
        "google.generativeai": module("google.generativeai"),
        "huggingface_hub": module("huggingface_hub", InferenceClient=Mock()),
        "datasets": module("datasets", Dataset=type("Dataset", (), {})),
        "tools": module("tools", ArxivSearch=Mock()),
        "inference": module("inference", query_model=Mock(), json=json),
        "app": module("app"),
        "PyPDF2": module("PyPDF2"),
        "common_imports": module("common_imports", os=os, re=re, time=time, json=json),
        "mlesolver": module("mlesolver", MLESolver=Mock()),
    }
    with patch.dict(sys.modules, dependencies):
        utils = importlib.import_module("utils")
        agents = importlib.import_module("agents")
        workflow = importlib.import_module("ai_lab_repo")
    # Keep class module identities available for saved-workflow pickle checks.
    sys.modules.update(utils=utils, agents=agents, ai_lab_repo=workflow)
    return workflow, agents


workflow, agents = load_workflow()


def make_lab(enabled=False, steps=3):
    lab = workflow.LaboratoryWorkflow("Find research resources", None,
                                     parallel_web_search=enabled, max_steps=steps,
                                     num_papers_lit_review=99,
                                     human_in_loop_flag={"literature review": False})
    lab.verbose = False
    return lab
