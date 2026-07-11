import os, re
import shutil
import time
import openai
import subprocess, string
from openai import OpenAI
from huggingface_hub import InferenceClient
from tokenization import count_message_tokens, get_encoding


def load_genai():
    import google.generativeai as genai

    return genai


def query_deepseekv3(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=False, temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Query qwen error: {e}")
        if attempt >= 10: return f"Your attempt to query deepseekv3 failed: {e}"
        return query_deepseekv3(prompt, system, api_key, attempt=attempt+1, temperature=temperature)


def query_qwen(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        client = InferenceClient(api_key=api_key)
        if system is not None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "user", "content": prompt}]

        completion = client.chat.completions.create(
            model="Qwen/QwQ-32B",
            messages=messages,
            max_tokens=500,
            temperature=temperature
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Query qwen error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        return query_qwen(prompt, system, api_key, attempt=attempt+1, temperature=temperature)


def query_gpt4omini(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        openai_api_key = api_key
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if system is not None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "user", "content": prompt}]
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=temperature).choices[0].message.content.strip()
        return response
    except Exception as e:
        print(f"Query 4o-mini error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        return query_gpt4omini(prompt, system, api_key, attempt=attempt+1, temperature=temperature)



def query_gpt4o(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        openai_api_key = api_key
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if system is not None:
            messages = [
                {"role": "user", "content":system + prompt}]
        else:
            messages = [
                {"role": "user", "content": prompt}]
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=temperature).choices[0].message.content.strip()
        return response
    except Exception as e:
        print(f"Query gpr-4o error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        return query_gpt4o(prompt, system, api_key, attempt=attempt+1, temperature=temperature)



def query_gemini(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        genai = load_genai()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system)
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature)).text.strip()
        time.sleep(1)
        return response
    except Exception as e:
        print(f"Gemini error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        time.sleep(1)
        return query_gemini(prompt, system, api_key, attempt=attempt+1, temperature=temperature)



def query_gemini2p0(prompt, system, api_key, attempt=0, temperature=0.0,):
    try:
        genai = load_genai()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash", system_instruction=system)
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature)).text.strip()
        time.sleep(1)
        return response
    except Exception as e:
        print(f"Gemini error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        time.sleep(1)
        return query_gemini2p0(prompt, system, api_key, attempt=attempt+1, temperature=temperature)


def compile_latex(latex_code, output_path, compile=True, timeout=30):
    latex_code = latex_code.replace(
        r"\documentclass{article}",
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{array}\n\\usepackage{algorithm}\n\\usepackage{algorithmicx}\n\\usepackage{algpseudocode}\n\\usepackage{booktabs}\n\\usepackage{colortbl}\n\\usepackage{color}\n\\usepackage{enumitem}\n\\usepackage{fontawesome5}\n\\usepackage{float}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{listings}\n\\usepackage{makecell}\n\\usepackage{multicol}\n\\usepackage{multirow}\n\\usepackage{pgffor}\n\\usepackage{pifont}\n\\usepackage{soul}\n\\usepackage{sidecap}\n\\usepackage{subcaption}\n\\usepackage{titletoc}\n\\usepackage[symbol]{footmisc}\n\\usepackage{url}\n\\usepackage{wrapfig}\n\\usepackage{xcolor}\n\\usepackage{xspace}")
    #print(latex_code)
    dir_path = f"{output_path}/tex"
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
        return f"[CODE EXECUTION ERROR]: Compilation failed. There was an error in your latex."


def count_tokens(messages, model="gpt-4"):
    return count_message_tokens(messages, model)

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


def clip_tokens(messages, model="gpt-4", max_tokens=100000):
    enc = get_encoding(model)
    total_tokens = count_message_tokens(messages, model)

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

from typing import Dict, List

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution"])),
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    indices = [pos for pos, char in enumerate(results[0]) if char == "$"]
    if len(indices) <= 1:
        answer = results[0]
    else:
        answer = results[0][indices[0] + 1 : indices[-1]]

    if is_equiv(answer, remove_boxed(last_boxed_only_string(doc["solution"]))):
        retval = 1

    results = {
        "exact_match": retval,
    }
    return results


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def clean_answer(s):
    s = s.replace("\\dfrac", "\\frac") # makes no difference but can lead to errors
    s = s.replace("x \\in", "")
    return s

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return clean_answer(s[len(left) : -1])


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"
    if string == "5.5":
        string = "\\frac{11}{2}"
    if "(x - 3)(x + 3)" in string:
        string = string.replace("(x - 3)(x + 3)", "(x+3)(x-3)")

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
