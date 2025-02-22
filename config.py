"""
Configuration Guide for API Base URLs

Note: You can set `OLLAMA_API_BASE_URL` to your Ollama API URL if you are using it.
      You still need to set the `api_key` to the `ollama` when using it. (The `OpenAI API Key` field in the UI)
      Because we use it to identify if we are using ollama providers
      Then you can set any model string in the `args.llm_backend` flag or the `Custom LLM Backend (For Ollama)` field in the UI.

Read more about Ollama: https://ollama.com/blog/openai-compatibility
"""
GOOGLE_GENERATIVE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1"
OLLAMA_API_BASE_URL = "http://localhost:11434/v1/"

"""
TASK_NOTE_LLM Configuration Guide

# Phase Configuration
- phases need to one of the following: 
- ["literature review", "plan formulation", 
   "data preparation", "running experiments", 
   "results interpretation", "report writing", 
   "report refinement"]

--- 
# Note Configuration
There are some variables that you can use in the note, you can use them by putting them in double curly braces.
Example: "You should write the report in {{language}}"

Here are the available variables for common use:
- research_topic: The research topic of the task
- api_key: OpenAI API Key
- deepseek_api_key: Deepseek API Key
- google_api_key: Google API Key
- anthropic_api_key: Anthropic API Key
- language: The language to use for the report
- llm_backend: The backend to use for the LLM
"""
TASK_NOTE_LLM = [
    {"phases": ["plan formulation"],
     "note": f"You should come up with a plan for TWO experiments."},

    {"phases": ["plan formulation", "data preparation", "running experiments"],
     "note": "Please use gpt-4o-mini for your experiments."},

    {"phases": ["running experiments"],
     "note": 'Use the following code to inference gpt-4o-mini: \nfrom openai import OpenAI\nos.environ["OPENAI_API_KEY"] = "{{api_key}}"\nclient = OpenAI()\ncompletion = client.chat.completions.create(\nmodel="gpt-4o-mini-2024-07-18", messages=messages)\nanswer = completion.choices[0].message.content\n'},

    {"phases": ["running experiments"],
     "note": "You have access to only gpt-4o-mini using the OpenAI API, please use the following key {{api_key}} but do not use too many inferences. Do not use openai.ChatCompletion.create or any openai==0.28 commands. Instead use the provided inference code."},

    {"phases": ["running experiments"],
     "note": "I would recommend using a small dataset (approximately only 100 data points) to run experiments in order to save time. Do not use much more than this unless you have to or are running the final tests."},

    {"phases": ["data preparation", "running experiments"],
     "note": "You are running on a Ubuntu System. You can use 'cuda' with PyTorch"},

    {"phases": ["data preparation", "running experiments"],
     "note": "Generate figures with very colorful and artistic design."},

    {"phases": ["literature review", "plan formulation",
                "data preparation", "running experiments",
                "results interpretation", "report writing",
                "report refinement"],
     "note": "You should always write in the following language to converse and to write the report {{language}}"}
]

"""
Human-in-the-Loop Configuration Guide

You can configure Stages where human input will be requested.
- If the value is `True`, the stage will be in human mode.
- If the value is `False`, the stage will be in AI mode.
- If you set to `None`, the stage will take the configuration from the `args.copilot_mode` flag. 
  (The `Enable Human-in-Loop Mode` checkbox in the UI)
"""
CONFIG_HUMAN_IN_THE_LOOP = {
    "literature review":      None,
    "plan formulation":       None,
    "data preparation":       None,
    "running experiments":    None,
    "results interpretation": None,
    "report writing":         None,
    "report refinement":      None,
}

"""
Agent Models Configuration Guide

You can configure the LLM Backend used for the different phases.
- If the value is a string, the stage will use the specified backend.
- If the value is `None`, the stage will take the configuration from the `args.llm_backend` flag.
  (Or whatever model you select or set in the UI)
"""
CONFIG_AGENT_MODELS = {
    "literature review":      None,
    "plan formulation":       None,
    "data preparation":       None,
    "running experiments":    None,
    "results interpretation": None,
    "report writing":         None,
    "report refinement":      None,
}
