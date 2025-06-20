# Agent Laboratory VM Execution Steps

This guide provides step-by-step instructions to set up and run the Agent Laboratory project's main program (`ai_lab_repo.py`) in a clean virtual machine (VM) environment.

## 1. Prerequisites

Before you begin, ensure the following software is installed on your VM:

*   **Python:** Version 3.12 or newer (as recommended by the project's README).
*   **pip:** Python package installer (usually comes with Python).
*   **virtualenv:** Tool to create isolated Python environments (install with `pip install virtualenv`).
*   **Git:** For cloning the repository.
*   **(Optional but Recommended for full functionality) `pdflatex`:** For compiling LaTeX reports. Installation instructions vary by OS (e.g., `sudo apt install texlive-latex-full` on Debian/Ubuntu). If not installed, PDF compilation can be disabled via YAML configuration (`compile-latex: "false"`).

## 2. Setup and Execution Steps

Follow these commands to set up the environment and run the program. All output from these steps (including installation and program execution) will be logged to `setup_and_run.log`.

```bash
# Clone the repository (if not already done)
# git clone https://github.com/SamuelSchmidgall/AgentLaboratory.git
# cd AgentLaboratory

# Start logging all subsequent commands in this session to setup_and_run.log
# (Note: The actual command redirection will be on the python execution line.
# This comment is for clarity that all relevant output should end up there.)

# Create a Python virtual environment
python3 -m venv agent_lab_env >> setup_and_run.log 2>&1

# Activate the virtual environment
source agent_lab_env/bin/activate >> setup_and_run.log 2>&1

# Upgrade pip
pip install --upgrade pip >> setup_and_run.log 2>&1

# Install dependencies from requirements.txt
pip install -r requirements.txt >> setup_and_run.log 2>&1

# Install missing dependencies (for AgentRxiv/app.py and other functionalities)
# Note: Versions are not specified for these in the original project; using pip to get compatible versions.
pip install Flask Flask-SQLAlchemy sentence-transformers google-generativeai PyPDF2 >> setup_and_run.log 2>&1

# Prepare for execution:
# You will need API keys for LLM providers (OpenAI, DeepSeek, Anthropic, Google Gemini).
# These can be set as environment variables (e.g., OPENAI_API_KEY) or specified in the YAML config.
# For this example, we assume they are set as environment variables.
# export OPENAI_API_KEY="your_openai_api_key_here"
# (Replace with actual keys or ensure the chosen YAML config does not require one you haven't set)

# Run the main program using a sample configuration.
# All output (stdout and stderr) from this command will be appended to setup_and_run.log.
# Make sure the YAML path is correct and it's configured for a simple, quick run if possible,
# or uses models/APIs you have access to.
# The MATH_agentlab.yaml is chosen as an example.
echo "Starting AgentLaboratory execution..." >> setup_and_run.log 2>&1
python ai_lab_repo.py --yaml-location "experiment_configs/MATH_agentlab.yaml" >> setup_and_run.log 2>&1
echo "AgentLaboratory execution finished." >> setup_and_run.log 2>&1

# Deactivate the virtual environment (optional, after checking logs)
# deactivate
```

**Important Notes for Execution:**

*   **API Keys:** The program requires API keys for the LLMs it uses. Ensure these are set as environment variables (e.g., `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`) or correctly configured within the YAML file specified in `--yaml-location`. If keys are missing for services the chosen config attempts to use, the program will error out.
*   **YAML Configuration:** The example uses `experiment_configs/MATH_agentlab.yaml`. This file might need adjustments:
    *   Set `copilot_mode: "true"` for interactive prompts if you don't want a fully autonomous run or if some steps require decisions. For a fully automated test, ensure it's `"false"`.
    *   Change `llm-backend` or specific phase models to ones you have API access for.
    *   To minimize runtime for a test, you might want to reduce `num-papers-lit-review`, `mlesolver-max-steps`, `papersolver-max-steps`, and `num-papers-to-write`.
    *   If `pdflatex` is not installed, ensure `compile-latex: "false"` is in the YAML.

## 3. Creating a Reproducible Dependency File

After successful installation and execution, you can create an exact list of all packages and their versions that were used. This is helpful for future reproducibility.

Run the following command while the virtual environment (`agent_lab_env`) is still active:

```bash
pip freeze > requirements_exact.txt
```

This will create a `requirements_exact.txt` file. This file will contain all the packages installed in the virtual environment, including the dependencies of the packages listed in the original `requirements.txt`, with their exact versions.

## 4. Verification

To verify that the setup and program execution were successful (or to diagnose issues):

1.  **Open the `setup_and_run.log` file.** This file contains:
    *   Output from the `pip install` commands. Look for messages like "Successfully installed..." for various packages. Check for any installation errors.
    *   Output from the `python ai_lab_repo.py ...` command. This will include:
        *   Log messages from Agent Laboratory indicating the start and progress through different research phases (e.g., "Beginning phase: literature review", "Beginning subtask: plan formulation").
        *   Print statements from the agents or solvers.
        *   Any error messages or tracebacks if the program encountered issues.
        *   A final message like "AgentLaboratory execution finished." (which we echoed to the log).

2.  **Check for expected output directories/files:** Depending on the configuration, the program creates output directories (e.g., `MATH_research_dir/`, `state_saves/`, `uploads/`). The presence of these and files within them (like generated reports or code) can also indicate a successful run. The `setup_and_run.log` might mention the creation of these files.

A successful run will typically show phase progression without critical error tracebacks from the Python script itself. Errors related to API key access or specific LLM model failures might still occur if not configured correctly, and these would be visible in the log.
