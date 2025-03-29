# Agent Laboratory: Using LLM Agents as Research Assistants


<p align="center">
  <img src="media/AgentLabLogo.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

<p align="center">
    【English | <a href="readme/README-chinese.md">中文</a> | <a href="readme/README-japanese.md">日本語</a> | <a href="readme/README-korean.md">한국어</a> | <a href="readme/README-filipino.md">Filipino</a> | <a href="readme/README-french.md">Français</a> | <a href="readme/README-slovak.md">Slovenčina</a> | <a href="readme/README-portugese.md">Português</a> | <a href="readme/README-spanish.md">Español</a> | <a href="readme/README-turkish.md">Türkçe</a> | <a href="readme/README-hindi.md">हिंदी</a> | <a href="readme/README-bengali.md">বাংলা</a> | <a href="readme/README-vietnamese.md">Tiếng Việt</a> | <a href="readme/README-russian.md">Русский</a> | <a href="readme/README-arabic.md">العربية</a> | <a href="readme/README-farsi.md">فارسی</a> | <a href="readme/README-italian.md">Italiano</a>】
</p>

<p align="center">
    【📝 <a href="https://arxiv.org/pdf/2501.04227">Paper</a> | 🌐 <a href="https://agentlaboratory.github.io/">Website</a> | 🌐 <a href="https://agentrxiv.github.io/">AgentRxiv Website</a> | 💻 <a href="https://github.com/SamuelSchmidgall/AgentLaboratory">Software</a> | 📰 <a href="https://agentlaboratory.github.io/#citation-ref">Citation</a>】
</p>

### News 
* [March/24/2025] 🎉 🎊 🎉 Now introducing **AgentRxiv**, a framework where autonomous research agents can upload, retrieve, and build on each other’s research. This allows agents to make cumulative progress on their research.

## 📖 Overview

- **Agent Laboratory** is an end-to-end autonomous research workflow meant to assist **you** as the human researcher toward **implementing your research ideas**. Agent Laboratory consists of specialized agents driven by large language models to support you through the entire research workflow—from conducting literature reviews and formulating plans to executing experiments and writing comprehensive reports. 
- This system is not designed to replace your creativity but to complement it, enabling you to focus on ideation and critical thinking while automating repetitive and time-intensive tasks like coding and documentation. By accommodating varying levels of computational resources and human involvement, Agent Laboratory aims to accelerate scientific discovery and optimize your research productivity.
<p align="center">
  <img src="media/AgentLab.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

- Agent Laboratory also supports **AgentRxiv**, a framework where autonomous research agents can upload, retrieve, and build on each other’s research. This allows agents to make cumulative progress on their research.

<p align="center">
  <img src="media/agentrxiv.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>


### 🔬 How does Agent Laboratory work?

- Agent Laboratory consists of three primary phases that systematically guide the research process: (1) Literature Review, (2) Experimentation, and (3) Report Writing. During each phase, specialized agents driven by LLMs collaborate to accomplish distinct objectives, integrating external tools like arXiv, Hugging Face, Python, and LaTeX to optimize outcomes. This structured workflow begins with the independent collection and analysis of relevant research papers, progresses through collaborative planning and data preparation, and results in automated experimentation and comprehensive report generation. Details on specific agent roles and their contributions across these phases are discussed in the paper.

<p align="center">
  <img src="media/AgentLabWF.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>


### 👾 Currently supported models

* **OpenAI**: o1, o1-preview, o1-mini, gpt-4o, o3-mini
* **DeepSeek**: deepseek-chat (deepseek-v3)

To select a specific llm set the flag `--llm-backend="llm_model"` for example `--llm-backend="gpt-4o"` or `--llm-backend="deepseek-chat"`. Please feel free to add a PR supporting new models according to your need!

## 🖥️ Installation

### Python venv option

* We recommend using python 3.12

1. **Clone the GitHub Repository**: Begin by cloning the repository using the command:
```bash
git clone git@github.com:SamuelSchmidgall/AgentLaboratory.git
```

2. **Set up and Activate Python Environment**
```bash
python -m venv venv_agent_lab
```
- Now activate this environment:
```bash
source venv_agent_lab/bin/activate
```

3. **Install required libraries**
```bash
pip install -r requirements.txt
```

4. **Install pdflatex [OPTIONAL]**
```bash
sudo apt install pdflatex
```
## Instructions for Windows Users

To have a TeX engine on Windows, you need to install the MikTeX software. Follow these steps:

1. Go to the [MikTeX download page](https://miktex.org/download).
2. Download the installer for your version of Windows.
3. Run the installer and follow the on-screen instructions to complete the installation.
4. Once installed, you can verify the installation by opening a command prompt and typing `latex --version`.

- This enables latex source to be compiled by the agents.
- **[IMPORTANT]** If this step cannot be run due to not having sudo access, pdf compiling can be turned off via running Agent Laboratory via setting the `--compile-latex` flag to false: `--compile-latex "false"`



5. **Now run Agent Laboratory!**

`python ai_lab_repo.py --yaml-location "experiment_configs/MATH_agentlab.yaml"`


### Co-Pilot mode

To run Agent Laboratory in copilot mode, simply set the copilot-mode flag in your yaml config to `"true"`

-----
## Tips for better research outcomes


#### [Tip #1] 📝 Make sure to write extensive notes! 📝

**Writing extensive notes is important** for helping your agent understand what you're looking to accomplish in your project, as well as any style preferences. Notes can include any experiments you want the agents to perform, providing API keys, certain plots or figures you want included, or anything you want the agent to know when performing research.

This is also your opportunity to let the agent know **what compute resources it has access to**, e.g. GPUs (how many, what type of GPU, how many GBs), CPUs (how many cores, what type of CPUs), storage limitations, and hardware specs.

In order to add notes, you must modify the task_notes_LLM structure inside of `ai_lab_repo.py`. Provided below is an example set of notes used for some of our experiments. 


```
task-notes:
  plan-formulation:
    - 'You should come up with a plan for only ONE experiment aimed at maximizing performance on the test set of MATH using prompting techniques.'
    - 'Please use gpt-4o-mini for your experiments'
    - 'You must evaluate on the entire 500 test questions of MATH'
  data-preparation:
    - 'Please use gpt-4o-mini for your experiments'
    - 'You must evaluate on the entire 500 test questions of MATH'
    - 'Here is a sample code you can use to load MATH\nfrom datasets import load_dataset\nMATH_test_set = load_dataset("HuggingFaceH4/MATH-500")["test"]'
...
```

--------

#### [Tip #2] 🚀 Using more powerful models generally leads to better research 🚀

When conducting research, **the choice of model can significantly impact the quality of results**. More powerful models tend to have higher accuracy, better reasoning capabilities, and better report generation. If computational resources allow, prioritize the use of advanced models such as o1-(mini/preview) or similar state-of-the-art large language models.

However, **it’s important to balance performance and cost-effectiveness**. While powerful models may yield better results, they are often more expensive and time-consuming to run. Consider using them selectively—for instance, for key experiments or final analyses—while relying on smaller, more efficient models for iterative tasks or initial prototyping.

When resources are limited, **optimize by fine-tuning smaller models** on your specific dataset or combining pre-trained models with task-specific prompts to achieve the desired balance between performance and computational efficiency.

-----

#### [Tip #3] ✅ You can load previous saves from checkpoints ✅

**If you lose progress, internet connection, or if a subtask fails, you can always load from a previous state.** All of your progress is saved by default in the `state_saves` variable, which stores each individual checkpoint. 

-----


#### [Tip #4] 🈯 If you are running in a language other than English 🈲

If you are running Agent Laboratory in a language other than English, no problem, just make sure to provide a language flag to the agents to perform research in your preferred language. Note that we have not extensively studied running Agent Laboratory in other languages, so be sure to report any problems you encounter.

For example, if you are running in Chinese set the language in the yaml:

`language:  "中文"`

----


#### [Tip #5] 🌟 There is a lot of room for improvement 🌟

There is a lot of room to improve this codebase, so if you end up making changes and want to help the community, please feel free to share the changes you've made! We hope this tool helps you!


## 📜 License

Source Code Licensing: Our project's source code is licensed under the MIT License. This license permits the use, modification, and distribution of the code, subject to certain conditions outlined in the MIT License.

## 📬 Contact

If you would like to get in touch, feel free to reach out to [sschmi46@jhu.edu](mailto:sschmi46@jhu.edu)

## Reference / Bibtex


### Agent Laboratory
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

### AgentRxiv
```bibtex
@misc{schmidgall2025agentrxiv,
      title={AgentRxiv: Towards Collaborative Autonomous Research}, 
      author={Samuel Schmidgall and Michael Moor},
      year={2025},
      eprint={2503.18102},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2503.18102}, 
}
```