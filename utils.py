import os, re
import shutil
import tiktoken
import subprocess


def compile_latex(latex_code, compile=True, output_filename="output.pdf", timeout=30):
    latex_code = latex_code.replace(
        r"\documentclass{article}",
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{array}\n\\usepackage{algorithm}\n\\usepackage{algorithmicx}\n\\usepackage{algpseudocode}\n\\usepackage{booktabs}\n\\usepackage{colortbl}\n\\usepackage{color}\n\\usepackage{enumitem}\n\\usepackage{fontawesome5}\n\\usepackage{float}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{listings}\n\\usepackage{makecell}\n\\usepackage{multicol}\n\\usepackage{multirow}\n\\usepackage{pgffor}\n\\usepackage{pifont}\n\\usepackage{soul}\n\\usepackage{sidecap}\n\\usepackage{subcaption}\n\\usepackage{titletoc}\n\\usepackage[symbol]{footmisc}\n\\usepackage{url}\n\\usepackage{wrapfig}\n\\usepackage{xcolor}\n\\usepackage{xspace}")
    #print(latex_code)
    dir_path = "research_dir/tex"
    tex_file_path = os.path.join(dir_path, "temp.tex")
    # Write the LaTeX code to the .tex file in the specified directory
    with open(tex_file_path, "w") as f:
        f.write(latex_code)

    if not compile:
        return f"Compilation successful"

    # Compiling the LaTeX code using pdflatex with non-interactive mode and timeout
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "temp.tex"],
            check=True,                   # Raises a CalledProcessError on non-zero exit codes
            stdout=subprocess.PIPE,        # Capture standard output
            stderr=subprocess.PIPE,        # Capture standard error
            timeout=timeout,               # Timeout for the process
            cwd=dir_path
        )

        # If compilation is successful, return the success message
        return f"Compilation successful: {result.stdout.decode('utf-8')}"

    except subprocess.TimeoutExpired:
        # If the compilation takes too long, return a timeout message
        return "[CODE EXECUTION ERROR]: Compilation timed out after {} seconds".format(timeout)
    except subprocess.CalledProcessError as e:
        # If there is an error during LaTeX compilation, return the error message
        return f"[CODE EXECUTION ERROR]: Compilation failed: {e.stderr.decode('utf-8')} {e.output.decode('utf-8')}. There was an error in your latex."


def count_tokens(messages, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    num_tokens = sum([len(enc.encode(message["content"])) for message in messages])
    return num_tokens

def remove_figures():
    """Remove a directory if it exists."""
    for _file in os.listdir("."):
        if "Figure_" in _file and ".png" in _file:
            os.remove(_file)

def remove_directory(dir_path):
    """Remove a directory if it exists."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} removed successfully.")
        except Exception as e:
            print(f"Error removing directory {dir_path}: {e}")
    else:
        print(f"Directory {dir_path} does not exist or is not a directory.")


def save_to_file(location, filename, data):
    """Utility function to save data as plain text."""
    filepath = os.path.join(location, filename)
    try:
        with open(filepath, 'w') as f:
            f.write(data)  # Write the raw string instead of using json.dump
        print(f"Data successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving file {filename}: {e}")


def clip_tokens(messages, model="o1-mini", max_tokens=128000):
    enc = tiktoken.encoding_for_model(model)
    total_tokens = sum([len(enc.encode(message["content"])) for message in messages])

    if total_tokens <= max_tokens:
        return messages  # No need to clip if under the limit

    # Start removing tokens from the beginning
    tokenized_messages = []
    for message in messages:
        tokenized_content = enc.encode(message["content"])
        tokenized_messages.append({"role": message["role"], "content": tokenized_content})

    # Flatten all tokens
    all_tokens = [token for message in tokenized_messages for token in message["content"]]

    # Remove tokens from the beginning
    clipped_tokens = all_tokens[total_tokens - max_tokens:]

    # Rebuild the clipped messages
    clipped_messages = []
    current_idx = 0
    for message in tokenized_messages:
        message_token_count = len(message["content"])
        if current_idx + message_token_count > len(clipped_tokens):
            clipped_message_content = clipped_tokens[current_idx:]
            clipped_message = enc.decode(clipped_message_content)
            clipped_messages.append({"role": message["role"], "content": clipped_message})
            break
        else:
            clipped_message_content = clipped_tokens[current_idx:current_idx + message_token_count]
            clipped_message = enc.decode(clipped_message_content)
            clipped_messages.append({"role": message["role"], "content": clipped_message})
            current_idx += message_token_count
    return clipped_messages



def extract_prompt(text, word):
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(code_blocks).strip()
    return extracted_code


