import os
import subprocess
import sys
from typing import Any
import webbrowser

import gradio as gr


def get_existing_saves() -> list:
    """Retrieve list of existing save files from state_saves directory."""
    saves_dir = 'state_saves'
    try:
        os.makedirs(saves_dir, exist_ok=True)
        # List all .pkl files in the directory
        saves = [f for f in os.listdir(saves_dir) if f.endswith('.pkl')]
        return saves if saves else ["No saved states found"]
    except Exception as e:
        print(f"Error retrieving saves: {e}")
        return ["No saved states found"]


def refresh_saves_dropdown():
    """
    IMPORTANT PART:
    Return a *new* gr.Dropdown component populated with fresh choices.
    This replaces the existing dropdown instead of attempting to update it.
    """
    new_saves = get_existing_saves()
    return gr.Dropdown(
        choices=new_saves,
        label="Select Saved Research State",
        interactive=True
    )


def run_research_process(
    research_topic: str,
    api_key: str,
    llm_backend: str,
    custom_llm_backend: str,
    ollama_max_tokens: Any,
    language: str,
    copilot_mode: bool,
    compile_latex: bool,
    num_papers_lit_review: Any,  # Gradio numbers may come as float
    mlesolver_max_steps: Any,
    papersolver_max_steps: Any,
    deepseek_api_key: str = "",
    google_api_key: str = "",
    anthropic_api_key: str = "",
    load_existing: bool = False,
    load_existing_path: str = ""
) -> str:
    # Determine which LLM backend to use:
    if api_key.strip().lower() == "ollama":
        chosen_backend = custom_llm_backend.strip() if custom_llm_backend.strip() else llm_backend
    else:
        chosen_backend = llm_backend

    # Prepare the command arguments
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

    # Add optional API keys if provided
    if api_key:
        cmd.extend(['--api-key', api_key])
    if deepseek_api_key:
        cmd.extend(['--deepseek-api-key', deepseek_api_key])
    if google_api_key:
        cmd.extend(['--google-api-key', google_api_key])
    if anthropic_api_key:
        cmd.extend(['--anthropic-api-key', anthropic_api_key])

    # Valid API keys are required for the research process to start
    if not api_key and not deepseek_api_key and not google_api_key and not anthropic_api_key:
        return "**Error starting research process:** No valid API key provided. At least one API key is required."

    # For Ollama, require a custom LLM backend and add the custom max tokens
    if api_key.strip().lower() == "ollama":
        if not custom_llm_backend.strip():
            return "**Error starting research process:** Custom LLM Backend is required for Ollama. Enter a custom model string or select a standard model."
        if not ollama_max_tokens:
            return "**Error starting research process:** Custom Max Tokens for Ollama is required. Enter a valid integer value."
        # Append custom max tokens for Ollama
        cmd.extend(['--ollama-max-tokens', str(int(ollama_max_tokens))])

    # Add load existing flags if selected
    if load_existing and load_existing_path and load_existing_path != "No saved states found":
        cmd.extend([
            '--load-existing', 'True',
            '--load-existing-path', os.path.join('state_saves', load_existing_path)
        ])

    # Create a string version of the command for display purposes.
    command_str = ' '.join([
        arg if (arg == sys.executable or arg == "ai_lab_repo.py" or arg.startswith("--"))
        else f'"{arg}"'
        for arg in cmd
    ])

    # Build the Markdown status message with the created command
    markdown_status = f"""**Command created:**  
<details>
    <summary>Click to view the command created</summary>
    <pre>{command_str}</pre>
</details>
"""

    # Now attempt to open a new terminal window with the research process
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


def create_gradio_config() -> gr.Blocks:
    # Standard backend options for the dropdown
    llm_backend_options = [
        "o1",
        "o1-preview",
        "o1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "deepseek-chat",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]
    languages = [
        "English", "Chinese-Simplified", "Chinese-Traditional",
        "Japanese", "Korean", "Filipino", "French",
        "Slovak", "Portuguese", "Spanish", "Turkish", "Hindi", "Bengali",
        "Vietnamese", "Russian", "Arabic", "Farsi", "Italian"
    ]

    with gr.Blocks() as demo:
        gr.Markdown("# Agent Laboratory Configuration")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Basic Configuration")
                research_topic = gr.Textbox(
                    label="Research Topic",
                    placeholder="Enter your research idea...",
                    lines=3
                )
                api_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="Enter your OpenAI API key (for Ollama, set API key to 'ollama', must be set)"
                )
                deepseek_api_key = gr.Textbox(
                    label="DeepSeek API Key (Optional)",
                    type="password",
                    placeholder="Enter your DeepSeek API key if using DeepSeek model"
                )
                google_api_key = gr.Textbox(
                    label="Google API Key (Optional)",
                    type="password",
                    placeholder="Enter your Google API key if using Google models"
                )
                anthropic_api_key = gr.Textbox(
                    label="Anthropic API Key (Optional)",
                    type="password",
                    placeholder="Enter your Anthropic API key if using Anthropic models"
                )

                with gr.Row():
                    # Dropdown for standard LLM backend options
                    llm_backend = gr.Dropdown(
                        choices=llm_backend_options,
                        label="LLM Backend",
                        value="o1-mini"
                    )
                    language = gr.Dropdown(
                        choices=languages,
                        label="Language",
                        value="English"
                    )

                # Custom LLM Backend textbox for Ollama.
                # This is optional and only used when API key is set to "ollama".
                custom_llm_backend = gr.Textbox(
                    label="Custom LLM Backend (For Ollama)",
                    placeholder="Enter your custom model string (optional)",
                    value=""
                )
                # Custom max tokens for Ollama
                ollama_max_tokens = gr.Number(
                    label="Custom Max Tokens for Ollama",
                    value=2048,
                    precision=0,
                    info="Set the maximum tokens for the Ollama model (only used if API key is 'ollama')"
                )

            with gr.Column():
                gr.Markdown("## Advanced Configuration")
                with gr.Accordion(label="Instructions for Use:", open=True):
                    gr.Markdown(
                        """
                        - Fill in the research configuration.
                        - Optionally load a previous research state.
                        - **For standard models:** Select the desired backend from the dropdown.
                        - **For Ollama:** Set the API key to `ollama` and, if needed, enter your custom model string and max tokens in the **Custom LLM Backend** and **Custom Max Tokens for Ollama** fields.
                        - If the custom field is left empty when using Ollama, the dropdown value will be used.
                        - Click **Start Research in Terminal**.
                        - A new terminal window will open with the research process.
                        """
                    )

                # Configuration for the research process
                with gr.Row():
                    copilot_mode = gr.Checkbox(label="Enable Human-in-Loop Mode")
                    compile_latex = gr.Checkbox(label="Compile LaTeX", value=True)

                with gr.Row():
                    num_papers_lit_review = gr.Number(
                        label="Papers in Literature Review",
                        value=5, precision=0, minimum=1, maximum=20
                    )
                    mlesolver_max_steps = gr.Number(
                        label="MLE Solver Max Steps",
                        value=3, precision=0, minimum=1, maximum=10
                    )
                    papersolver_max_steps = gr.Number(
                        label="Paper Solver Max Steps",
                        value=5, precision=0, minimum=1, maximum=10
                    )

                # Saved States Section
                with gr.Accordion("Resume Previous Research", open=False):
                    load_existing = gr.Checkbox(label="Load Existing Research State")
                    existing_saves = gr.Dropdown(
                        choices=get_existing_saves(),
                        label="Select Saved Research State",
                        interactive=True
                    )
                    refresh_saves_btn = gr.Button("Refresh Saved States")

        submit_btn = gr.Button("Start Research in Terminal", variant="primary")

        with gr.Accordion(label="Status", open=True):
            # Connect submit button to the research process.
            # Output is gr.Markdown so that the returned Markdown is rendered.
            submit_btn.click(
                fn=run_research_process,
                inputs=[
                    research_topic, api_key, llm_backend, custom_llm_backend, ollama_max_tokens, language,
                    copilot_mode, compile_latex, num_papers_lit_review,
                    mlesolver_max_steps, papersolver_max_steps,
                    deepseek_api_key, google_api_key, anthropic_api_key,
                    load_existing, existing_saves,
                ],
                outputs=gr.Markdown()
            )

        # Instead of returning just a list, return a new Dropdown from refresh_saves_dropdown()
        refresh_saves_btn.click(
            fn=refresh_saves_dropdown,
            inputs=None,
            outputs=existing_saves
        )

    return demo


def main():
    demo = create_gradio_config()
    demo.launch()


if __name__ == "__main__":
    main()
