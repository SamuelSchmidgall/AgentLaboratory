import os
import subprocess
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from settings_manager import SettingsManager

# Check if the WebUI repository is cloned
def check_webui_cloned():
    if not os.path.exists("AgentLaboratoryWebUI"):
        # Ask the user if they want to clone the repository
        print("The WebUI repository is not cloned.")
        answer = input("Would you like to clone it now from https://github.com/whats2000/AgentLaboratoryWebUI.git? (y/n) ")

        if answer.lower() != "y":
            print("Error: The WebUI repository is not cloned. Please clone it manually.")
            sys.exit(1)

        print("Cloning the WebUI repository...")
        subprocess.run(["git", "clone", "https://github.com/whats2000/AgentLaboratoryWebUI.git"], check=True)

# Check if Node.js is installed
def check_node_installed():
    try:
        subprocess.run(["node", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Error: Node.js is not installed. Please install it from https://nodejs.org/")
        sys.exit(1)

# Check if Yarn is installed
def check_yarn_installed():
    try:
        subprocess.run(["yarn", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        # Ask the user if they want to install Yarn
        print("Yarn is not installed.")
        answer = input("Would you like to install it now? (y/n) ")

        if answer.lower() != "y":
            print("Error: Yarn is not installed. Please install it manually.")
            sys.exit(1)

        print("Installing Yarn...")
        subprocess.run(["npm", "install", "-g", "yarn"], check=True)

# Build the WebUI
def build_webui():
    webui_path = os.path.join(os.getcwd(), "AgentLaboratoryWebUI")
    if not os.path.exists(os.path.join(webui_path, "dist")):
        print("Building the WebUI...")
        subprocess.run(["yarn", "install"], check=True, cwd=webui_path)
        subprocess.run(["yarn", "build"], check=True, cwd=webui_path)

# Run the checks and build the WebUI
check_webui_cloned()
check_node_installed()
check_yarn_installed()
build_webui()

# Initialize the Flask app
app = Flask(
    __name__,
    static_url_path='',
    static_folder='AgentLaboratoryWebUI/dist',
    template_folder='AgentLaboratoryWebUI/dist'
)
CORS(app)

# Define default values
DEFAULT_SETTINGS = {
    "research_topic": "",
    "api_key": "",
    "deepseek_api_key": "",
    "google_api_key": "",
    "anthropic_api_key": "",
    "llm_backend": "o1-mini",
    "custom_llm_backend": "",
    "ollama_max_tokens": 2048,
    "language": "English",
    "copilot_mode": False,
    "compile_latex": True,
    "num_papers_lit_review": 5,
    "mlesolver_max_steps": 3,
    "papersolver_max_steps": 5,
}

settings_manager = SettingsManager()

def save_user_settings_from_dict(settings: dict):
    """Save settings using the SettingsManager."""
    settings_manager.save_settings(settings)

def load_user_settings():
    """Load settings using the SettingsManager."""
    return settings_manager.load_settings()

def get_existing_saves():
    """
    Retrieve list of existing save files from the 'state_saves' directory.
    """
    saves_dir = 'state_saves'
    try:
        os.makedirs(saves_dir, exist_ok=True)
        saves = [f for f in os.listdir(saves_dir) if f.endswith('.pkl')]
        return saves if saves else ["No saved states found"]
    except Exception as e:
        print(f"Error retrieving saves: {e}")
        return ["No saved states found"]

def run_research_process(data: dict) -> str:
    """
    Execute the research process based on the provided settings.
    This is adapted from your original function.
    """
    # Unpack parameters from the incoming JSON payload.
    research_topic        = data.get('research_topic', '')
    api_key               = data.get('api_key', '')
    llm_backend           = data.get('llm_backend', 'o1-mini')
    custom_llm_backend    = data.get('custom_llm_backend', '')
    ollama_max_tokens     = data.get('ollama_max_tokens', 2048)
    language              = data.get('language', 'English')
    copilot_mode          = data.get('copilot_mode', False)
    compile_latex         = data.get('compile_latex', True)
    num_papers_lit_review = data.get('num_papers_lit_review', 5)
    mlesolver_max_steps   = data.get('mlesolver_max_steps', 3)
    papersolver_max_steps = data.get('papersolver_max_steps', 5)
    deepseek_api_key      = data.get('deepseek_api_key', '')
    google_api_key        = data.get('google_api_key', '')
    anthropic_api_key     = data.get('anthropic_api_key', '')
    load_existing         = data.get('load_existing', False)
    load_existing_path    = data.get('load_existing_path', '')

    # Choose backend based on the API key value.
    if api_key.strip().lower() == "ollama":
        chosen_backend = custom_llm_backend.strip() if custom_llm_backend.strip() else llm_backend
    else:
        chosen_backend = llm_backend

    # Prepare the command arguments.
    cmd = [
        sys.executable, 'ai_lab_repo.py',
        '--research-topic', research_topic,
        '--llm-backend', chosen_backend,
        '--language', language,
        '--copilot-mode', str(copilot_mode).lower(),
        '--compile-latex', str(compile_latex).lower(),
        '--num-papers-lit-review', str(num_papers_lit_review),
        '--mlesolver-max-steps', str(mlesolver_max_steps),
        '--papersolver-max-steps', str(papersolver_max_steps)
    ]

    # Append optional API keys if provided.
    if api_key:
        cmd.extend(['--api-key', api_key])
    if deepseek_api_key:
        cmd.extend(['--deepseek-api-key', deepseek_api_key])
    if google_api_key:
        cmd.extend(['--google-api-key', google_api_key])
    if anthropic_api_key:
        cmd.extend(['--anthropic-api-key', anthropic_api_key])

    # Require at least one valid API key.
    if not (api_key or deepseek_api_key or google_api_key or anthropic_api_key):
        return "**Error starting research process:** No valid API key provided. At least one API key is required."

    # Handle Ollama-specific requirements.
    if api_key.strip().lower() == "ollama":
        if not custom_llm_backend.strip():
            return "**Error starting research process:** Custom LLM Backend is required for Ollama. Enter a custom model string or select a standard model."
        if not ollama_max_tokens:
            return "**Error starting research process:** Custom Max Tokens for Ollama is required. Enter a valid integer value."
        cmd.extend(['--ollama-max-tokens', str(int(ollama_max_tokens))])

    # If loading an existing research state, add the flags.
    if load_existing and load_existing_path and load_existing_path != "No saved states found":
        cmd.extend([
            '--load-existing', 'True',
            '--load-existing-path', os.path.join('state_saves', load_existing_path)
        ])

    # Create a displayable command string.
    command_str = ' '.join(
        [arg if (arg == sys.executable or arg == "ai_lab_repo.py" or arg.startswith("--"))
         else f'"{arg}"'
         for arg in cmd]
    )
    markdown_status = f"**Command created:**\n```\n{command_str}\n```\n"

    # Attempt to open a new terminal window and run the command.
    try:
        if sys.platform == 'win32':
            subprocess.Popen(['start', 'cmd', '/k'] + cmd, shell=True)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', '-a', 'Terminal'] + cmd)
        else:
            subprocess.Popen(['x-terminal-emulator', '-e'] + cmd)
        markdown_status += "\n**Research process started in a new terminal window.**"
    except Exception as e:
        markdown_status += f"\n**Error starting research process:** {e}"

    return markdown_status

@app.route("/")
def hello():
    return render_template("index.html")

# Endpoint to start the research process.
@app.route('/api/research', methods=['POST'])
def api_research():
    data = request.get_json()
    result = run_research_process(data)
    return jsonify({"status": result})

# Endpoint to load or update settings.
@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'GET':
        settings = load_user_settings()
        # Merge with defaults to ensure all keys are present.
        merged_settings = DEFAULT_SETTINGS.copy()
        merged_settings.update(settings or {})
        return jsonify(merged_settings)
    elif request.method == 'POST':
        settings = request.get_json()
        save_user_settings_from_dict(settings)
        return jsonify({"status": "Settings saved"})

# Endpoint to retrieve saved research states.
@app.route('/api/saves', methods=['GET'])
def api_saves():
    saves = get_existing_saves()
    return jsonify({"saves": saves})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
