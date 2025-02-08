import os
import sys
import gradio as gr
import subprocess
import webbrowser

def get_existing_saves():
    """Retrieve list of existing save files from state_saves directory."""
    saves_dir = 'state_saves'
    try:
        # Ensure directory exists
        os.makedirs(saves_dir, exist_ok=True)
        
        # List all .pkl files in the directory
        saves = [f for f in os.listdir(saves_dir) if f.endswith('.pkl')]
        return saves if saves else ["No saved states found"]
    except Exception as e:
        print(f"Error retrieving saves: {e}")
        return ["No saved states found"]

def run_research_process(
    research_topic,
    api_key,
    llm_backend,
    language,
    copilot_mode,
    compile_latex,
    num_papers_lit_review,
    mlesolver_max_steps,
    papersolver_max_steps,
    deepseek_api_key="",
    load_existing=False,
    load_existing_path=""
):
    # Prepare the command arguments
    cmd = [
        sys.executable, 'ai_lab_repo.py',
        '--research-topic', research_topic,
        '--api-key', api_key,
        '--llm-backend', llm_backend,
        '--language', language,
        '--copilot-mode', str(copilot_mode).lower(),
        '--compile-latex', str(compile_latex).lower(),
        '--num-papers-lit-review', str(num_papers_lit_review),
        '--mlesolver-max-steps', str(mlesolver_max_steps),
        '--papersolver-max-steps', str(papersolver_max_steps)
    ]

    # Add DeepSeek API key if provided
    if deepseek_api_key:
        cmd.extend(['--deepseek-api-key', deepseek_api_key])

    # Add load existing flags if selected
    if load_existing and load_existing_path and load_existing_path != "No saved states found":
        cmd.extend([
            '--load-existing', 'True',
            '--load-existing-path', os.path.join('state_saves', load_existing_path)
        ])

    # Open a new terminal window with the research process
    if sys.platform == 'win32':
        # Windows
        subprocess.Popen(['start', 'cmd', '/k'] + cmd, shell=True)
    elif sys.platform == 'darwin':
        # macOS
        subprocess.Popen(['open', '-a', 'Terminal'] + cmd)
    else:
        # Linux and other Unix-like systems
        subprocess.Popen(['x-terminal-emulator', '-e'] + cmd)

    return "Research process started in a new terminal window. Please check the terminal for progress."

def create_gradio_config():
    # Define Gradio interface
    llm_backends = ["o1", "o1-preview", "o1-mini", "gpt-4o", "gpt-4o-mini", "deepseek-chat"]
    languages = ["English", "Chinese-Simplified", "Chinese-Traditional",
                 "Japanese", "Korean", "Filipino", "French",
                 "Slovak", "Portuguese", "Spanish", "Turkish", "Hindi", "Bengali",
                 "Vietnamese", "Russian", "Arabic", "Farsi", "Italian"]

    with gr.Blocks() as demo:
        gr.Markdown("# Agent Laboratory Configuration")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Research Configuration")
                research_topic = gr.Textbox(label="Research Topic", placeholder="Enter your research idea...", lines=3)
                api_key = gr.Textbox(label="OpenAI API Key", type="password", placeholder="Enter your OpenAI API key")
                deepseek_api_key = gr.Textbox(label="DeepSeek API Key (Optional)", type="password", placeholder="Enter your DeepSeek API key if using DeepSeek model")

                with gr.Row():
                    llm_backend = gr.Dropdown(llm_backends, label="LLM Backend", value="o1-mini")
                    language = gr.Dropdown(languages, label="Language", value="English")

                with gr.Row():
                    copilot_mode = gr.Checkbox(label="Enable Human-in-Loop Mode")
                    compile_latex = gr.Checkbox(label="Compile LaTeX", value=True)

                with gr.Row():
                    num_papers_lit_review = gr.Number(label="Papers in Literature Review", value=5, precision=0, minimum=1, maximum=20)
                    mlesolver_max_steps = gr.Number(label="MLE Solver Max Steps", value=3, precision=0, minimum=1, maximum=10)
                    papersolver_max_steps = gr.Number(label="Paper Solver Max Steps", value=5, precision=0, minimum=1, maximum=10)

                # Saved States Section
                with gr.Accordion("Resume Previous Research", open=False):
                    load_existing = gr.Checkbox(label="Load Existing Research State")
                    existing_saves = gr.Dropdown(
                        choices=get_existing_saves(), 
                        label="Select Saved Research State", 
                        interactive=True
                    )
                    # Refresh button for saves
                    refresh_saves_btn = gr.Button("Refresh Saved States")

                submit_btn = gr.Button("Start Research in Terminal", variant="primary")

            with gr.Column():
                gr.Markdown("## Instructions")
                gr.Markdown("""
                - Fill in the research configuration
                - Optionally load a previous research state
                - Click 'Start Research in Terminal'
                - A new terminal window will open with the research process
                """)

        # Refresh saved states
        refresh_saves_btn.click(
            fn=get_existing_saves,
            outputs=existing_saves
        )

        # Connect submit button to research process
        submit_btn.click(
            run_research_process,
            inputs=[
                research_topic, api_key, llm_backend, language,
                copilot_mode, compile_latex, num_papers_lit_review,
                mlesolver_max_steps, papersolver_max_steps,
                deepseek_api_key, 
                load_existing,
                existing_saves,
            ],
            outputs=gr.Textbox(label="Status")
        )

    return demo

def main():
    demo = create_gradio_config()
    # Run on all network interfaces, port 7860
    demo.launch()

if __name__ == "__main__":
    main()