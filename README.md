# Agent Laboratory: Using LLM Agents as Research Assistants


<p align="center">
  <img src="media/AgentLabLogo.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

<p align="center">
    【English | <a href="readme/README-chinese.md">中文</a> | <a href="readme/README-japanese.md">日本語</a> | <a href="readme/README-korean.md">한국어</a> | <a href="readme/README-filipino.md">Filipino</a> | <a href="readme/README-french.md">Français</a> | <a href="readme/README-slovak.md">Slovenčina</a> | <a href="readme/README-portugese.md">Português</a> | <a href="readme/README-spanish.md">Español</a> | <a href="readme/README-turkish.md">Türkçe</a> | <a href="readme/README-hindi.md">हिंदी</a> | <a href="readme/README-bengali.md">বাংলা</a> | <a href="readme/README-vietnamese.md">Tiếng Việt</a> | <a href="readme/README-russian.md">Русский</a> | <a href="readme/README-arabic.md">العربية</a> | <a href="readme/README-farsi.md">فارسی</a> | <a href="readme/README-italian.md">Italiano</a>】
</p>

<p align="center">
    【📝 <a href="https://arxiv.org/pdf/2501.04227">Paper</a> | 🌐 <a href="https://agentlaboratory.github.io/">Website</a> | 💻 <a href="https://github.com/SamuelSchmidgall/AgentLaboratory">Software</a> | 📰 <a href="https://agentlaboratory.github.io/#citation-ref">Citation</a>】
</p>

## 📖 Overview

- **Agent Laboratory** is an end-to-end autonomous research workflow meant to assist **you** as the human researcher toward **implementing your research ideas**. Agent Laboratory consists of specialized agents driven by large language models to support you through the entire research workflow—from conducting literature reviews and formulating plans to executing experiments and writing comprehensive reports. 
- This system is not designed to replace your creativity but to complement it, enabling you to focus on ideation and critical thinking while automating repetitive and time-intensive tasks like coding and documentation. By accommodating varying levels of computational resources and human involvement, Agent Laboratory aims to accelerate scientific discovery and optimize your research productivity.

<p align="center">
  <img src="media/AgentLab.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

### 🔬 How does Agent Laboratory work?

- Agent Laboratory consists of three primary phases that systematically guide the research process: (1) Literature Review, (2) Experimentation, and (3) Report Writing. During each phase, specialized agents driven by LLMs collaborate to accomplish distinct objectives, integrating external tools like arXiv, Hugging Face, Python, and LaTeX to optimize outcomes. This structured workflow begins with the independent collection and analysis of relevant research papers, progresses through collaborative planning and data preparation, and results in automated experimentation and comprehensive report generation. Details on specific agent roles and their contributions across these phases are discussed in the paper.

<p align="center">
  <img src="media/AgentLabWF.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>


### 👾 Currently supported models

* **OpenAI**: o1, o1-preview, o1-mini, gpt-4o
* **DeepSeek**: deepseek-chat (deepseek-v3)
* **Anthropic**: claude-3-5-sonnet, claude-3-5-haiku
* **Google**: gemini-2.0-flash, gemini-2.0-flash
* **Ollama**: Any model that you can find in the [Ollama Website](https://ollama.com/search)

To select a specific llm set the flag `--llm-backend="llm_model"` for example `--llm-backend="gpt-4o"` or `--llm-backend="deepseek-chat"`. Please feel free to add a PR supporting new models according to your need!

## 🖥️ Installation

> [!IMPORTANT]
> We recommend using python 3.12

1. **Clone the GitHub Repository**: Begin by cloning the repository using the command:
    ```bash
    git clone git@github.com:SamuelSchmidgall/AgentLaboratory.git
    ```

2. **Set up and Activate Python Environment**
    
    Python venv option
    ```bash
    python -m venv venv_agent_lab
    source venv_agent_lab/bin/activate
    ```
   
    Conda option
    ```bash
    conda create -n agent_lab python=3.12
    conda activate agent_lab
    ```

3. **Install required libraries**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install Higher Version of Gradio**
    ```bash
    pip install gradio==4.44.1
    ```

> [!NOTE]
> This is only required for the current version of web interface. 
> We will move to `Flask` in the future for capability of the package.

5. **Install pdflatex [OPTIONAL]**

    For Ubuntu:
    ```bash
    sudo apt install pdflatex
    ```
    
    If you find the package is not available, 
    you can install it via the following commands:
    ```bash
    sudo apt-get install texlive-latex-base
    
    sudo apt-get install texlive-fonts-recommended
    sudo apt-get install texlive-fonts-extra
    
    sudo apt-get install texlive-latex-extra
    ```
    
    - This enables latex source to be compiled by the agents.
> [!IMPORTANT] 
> If this step cannot be run due to not having sudo access, 
pdf compiling can be turned off via running Agent Laboratory 
via setting the `--compile-latex` flag to false: `--compile-latex "false"`.
Or you can disable by unchecked the `Compile LaTeX` option in the web interface.

## 🚀 Quick Start

1. **Set up the configuration file**

   - You can set up the configuration file by editing the `config.py` file.
   - See the [configuration file](./config.py) for more details.

2. **Now run Agent Laboratory!**

    #### Basic Usage of Agent Laboratory in Web Interface
    ```bash
    python config_gradio.py
    ```
    
    #### Basic Usage of Agent Laboratory in CLI

    ##### 1. A simple command to run Agent Laboratory
    ```bash
    python ai_lab_repo.py --api-key "API_KEY_HERE" --llm-backend "o1-mini" --research-topic "YOUR RESEARCH IDEA"
    ```
    
    ##### 2. Available Configuration Options

    **API Keys:**
    - `--api-key`: OpenAI API key or set to "ollama" for Ollama usage **(required)**
    - `--deepseek-api-key`: DeepSeek API key 
    - `--google-api-key`: Google API key
    - `--anthropic-api-key`: Anthropic API key

    **LLM Settings:**
    - `--llm-backend`: Backend LLM to use (default: "o1-mini"), please ensure your model string is correct, here is some common models:
      - OpenAI: "o1", "o1-preview", "o1-mini", "gpt-4o"
      - DeepSeek: "deepseek-chat" (deepseek-v3)
      - Anthropic: "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"
      - Google: "gemini-2.0-flash", "gemini-2.0-flash"
      - Ollama: Any model that you can find in the [Ollama Website](https://ollama.com/search)
    - `--ollama-max-tokens`: Max tokens for OLLAMA (default: 2048), 

    **Research Parameters:**
    - `--research-topic`: Your research topic/idea or a open-ended question to ask, this **must be provided**
    - `--language`: Operating language (default: "English") which will instruct the agents to perform research in your preferred language (Not fully supported yet)
    - `--num-papers-lit-review`: Number of papers for literature review (default: 5)
    - `--mlesolver-max-steps`: Steps for MLE solver (default: 3)
    - `--papersolver-max-steps`: Steps for paper solver (default: 5)
    
    **Operation Modes:**
    - `--copilot-mode`: Enable human interaction mode (default: "false"), you need check terminal for input in this mode
    - `--compile-latex`: Enable LaTeX PDF compilation (default: "true"), **please ensure you have pdflatex installed**
    
    **State Management:**
    - `--load-existing`: Load from existing state (default: "false")
    - `--load-existing-path`: Path to load state from (e.g., "state_saves/results_interpretation.pkl")

    <details>
    <summary>📚 Example Usage</summary>

    Basic run without PDF compilation:
    ```bash
    python ai_lab_repo.py --api-key "API_KEY_HERE" --llm-backend "o1-mini" --research-topic "YOUR RESEARCH IDEA" --compile-latex "false"
    ```
    
    Run in copilot mode:
    ```bash 
    python ai_lab_repo.py --api-key "API_KEY_HERE" --llm-backend "o1-mini" --research-topic "YOUR RESEARCH IDEA" --copilot-mode "true"
    ```
    
    Run with custom solver steps and language:
    ```bash
    python ai_lab_repo.py --api-key "API_KEY_HERE" --llm-backend "o1-mini" --research-topic "YOUR RESEARCH IDEA" --mlesolver-max-steps "5" --papersolver-max-steps "7" --language "Spanish"
    ```
    
    Load from the existing state:
    ```bash
    python ai_lab_repo.py --api-key "API_KEY_HERE" --load-existing "true" --research-topic "YOUR RESEARCH IDEA" --load-existing-path "state_saves/results_interpretation.pkl"
    ```
    </details>

> [!NOTE]
> You must at least provide an API key for use. 
> Even when you run a local Ollama, you must provide an "ollama" string as the API key.

> [!TIP]
> - Set the `--ollama-max-tokens` to the model real context length (Ex: 128000 for `qwen2.5:32b`) for much better performance.
> - Use the model that supports `tools` as the Agent Laboratory will instruct the model to output formatted code or actions (This is kinda needed for the current version of Agent Laboratory).

-----

## Tips for better research outcomes


#### [Tip #1] 📝 Make sure to write extensive notes! 📝

**Writing extensive notes is important** for helping your agent understand what you're looking to accomplish in your project, 
as well as any style preferences. Notes can include any experiments you want the agents to perform, providing API keys, certain plots or figures you want included, or anything you want the agent to know when performing research.

This is also your opportunity to let the agent know **what compute resources it has access to**, 
e.g. GPUs (how many, what type of GPU, how many GBs), CPUs (how many cores, what type of CPUs), storage limitations, and hardware specs.

In order to add notes, you must modify the TASK_NOTE_LLM structure inside of `config.py`. 
Provided below is an example set of notes used for some of our experiments. 


```
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
```

--------

#### [Tip #2] 🚀 Using more powerful models generally leads to better research 🚀

When conducting research, **the choice of model can significantly impact the quality of results**. More powerful models tend to have higher accuracy, better reasoning capabilities, and better report generation. If computational resources allow, prioritize the use of advanced models such as o1-(mini/preview) or similar state-of-the-art large language models.

However, **it’s important to balance performance and cost-effectiveness**. While powerful models may yield better results, they are often more expensive and time-consuming to run. Consider using them selectively—for instance, for key experiments or final analyses—while relying on smaller, more efficient models for iterative tasks or initial prototyping.

When resources are limited, **optimize by fine-tuning smaller models** on your specific dataset or combining pre-trained models with task-specific prompts to achieve the desired balance between performance and computational efficiency.

-----

#### [Tip #3] ✅ You can load previous saves from checkpoints ✅

**If you lose progress, internet connection, or if a subtask fails, you can always load from a previous state.** 
All of your progress is saved by default in the `state_saves` variable, which stores each individual checkpoint. 

##### **For Web Interface**

You can check out the `Resume Previous Research` section to load from a previous state.
By checking the `Load Existing Research State` flag and then select the stage you want to load from, you can easily load from a previous state.
If the state is not up-to-date, you can always click the `Refresh Saved States` button to refresh the saved states.

##### **For CLI**

Just pass the following arguments when running `ai_lab_repo.py`

```bash
python ai_lab_repo.py --api-key "API_KEY_HERE" --research-topic "YOUR RESEARCH IDEA" --llm-backend "o1-mini" --load-existing True --load-existing-path "state_saves/LOAD_PATH"
```

-----



#### [Tip #4] 🈯 If you are running in a language other than English 🈲

If you are running Agent Laboratory in a language other than English, no problem, just make sure to provide a language flag to the agents to perform research in your preferred language. Note that we have not extensively studied running Agent Laboratory in other languages, so be sure to report any problems you encounter.

##### **For Web Interface**
You can select the language in the dropdown menu. If the language you want is not available, you can edit the `config_gradio.py` file to add the language you want.

##### **For CLI**
If you are running in Chinese, you can run the following command:
`python ai_lab_repo.py --api-key "API_KEY_HERE" --research-topic "YOUR RESEARCH IDEA (in your language)" --llm-backend "o1-mini" --language "中文"`

----


#### [Tip #5] 🌟 There is a lot of room for improvement 🌟

There is a lot of room to improve this codebase, so if you end up making changes and want to help the community, please feel free to share the changes you've made! We hope this tool helps you!


## 📜 License

Source Code Licensing: Our project's source code is licensed under the MIT License. This license permits the use, modification, and distribution of the code, subject to certain conditions outlined in the MIT License.

## 📬 Contact

If you would like to get in touch, feel free to reach out to [sschmi46@jhu.edu](mailto:sschmi46@jhu.edu)

## Reference / Bibtex



```bibtex
@misc{schmidgall2025agentlaboratoryusingllm,
      title={Agent Laboratory: Using LLM Agents as Research Assistants}, 
      author={Samuel Schmidgall and Yusheng Su and Ze Wang and Ximeng Sun and Jialian Wu and Xiaodong Yu and Jiang Liu and Zicheng Liu and Emad Barsoum},
      year={2025},
      eprint={2501.04227},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2501.04227}, 
}
```
