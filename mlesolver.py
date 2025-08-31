import random
from copy import copy
from copy import deepcopy
from common_imports import *
from abc import abstractmethod


from tools import *
from inference import *
from pathlib import Path


from contextlib import contextmanager
import sys, os


os.environ["JOBLIB_VERBOSITY"] = "0"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logging.getLogger('sklearn.model_selection').setLevel(logging.WARNING)


GLOBAL_REPAIR_ATTEMPTS = 2


class Command:
    def __init__(self):
        self.cmd_type = "OTHER"

    @abstractmethod
    def docstring(self) -> str:
        pass

    @abstractmethod
    def execute_command(self, *args) -> str:
        pass

    @abstractmethod
    def matches_command(self, cmd_str) -> bool:
        pass

    @abstractmethod
    def parse_command(self, cmd_str) -> tuple:
        pass


"""
@@@@@@@@@@@@@@@@@@
@@ CODING TOOLS @@
@@@@@@@@@@@@@@@@@@
"""

class Replace(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-replace"

    def docstring(self) -> str:
        return (
            "============= REWRITE CODE EDITING TOOL =============\n"
            "You also have access to a code replacing tool. \n"
            "This tool allows you to entirely re-write/replace all of the current code and erase all existing code.\n"
            "You can use this tool via the following command: ```REPLACE\n<code here>\n```, where REPLACE is the word REPLACE and <code here> will be the new code that is replacing the entire set of old code. This tool is useful if you want to make very significant changes, such as entirely changing the model, or the learning process. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code. Try limiting the use of rewriting and aim for editing the code more."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> new code
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str) -> bool:
        if "```REPLACE" in cmd_str: return True
        return False

    def parse_command(self, *args) -> tuple:
        new_code = extract_prompt(args[0], "REPLACE")
        code_exec = f"{args[1]}\n{new_code}"
        code_ret = execute_code(code_exec)
        if "[CODE EXECUTION ERROR]" in code_ret: return False, (None, code_ret,)
        return True, (new_code.split("\n"), code_ret)



class Edit(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-edit"

    def docstring(self) -> str:
        return (
            "============= CODE EDITING TOOL =============\n"
            "You also have access to a code editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with code. \n"
            "You can edit code using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new code that is replacing the old code. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code. Your changes should significantly change the functionality of the code."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> N (int)
        # args[1] -> M (int)
        # args[2] -> old code
        # args[3] -> new lines to replace
        # args[4] -> new lines to replace
        try:
            args = args[0]
            current_code = args[2]
            lines_to_add = list(reversed(args[3]))
            lines_to_replace = list(reversed(range(args[0], args[1]+1)))
            for _ln in lines_to_replace:
                current_code.pop(_ln)
            for _line in lines_to_add:
                current_code.insert(args[0], _line)
            new_code = "\n".join(current_code)
            code_exec = f"{args[4]}\n{new_code}"
            code_ret = execute_code(code_exec)
            if "CODE EXECUTION ERROR" in code_ret: return (False, None, code_ret)
            return (True, current_code, code_ret)
        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        if "```EDIT" in cmd_str: return True
        return False

    def parse_command(self, *args) -> tuple:
        cmd_str, codelines, datasetcode = args[0], args[1], args[2]
        success = True
        try:
            text = extract_prompt(cmd_str, "EDIT").split("\n")
            if len(text) == 0: return False, None
            lines_to_edit = text[0].split(" ")
            if len(lines_to_edit) != 2: return False, None
            lines_to_edit = [int(_) for _ in lines_to_edit]
            if len(text[1:]) == 0: return False, None
            return success, (lines_to_edit[0], lines_to_edit[1], codelines, text[1:], datasetcode)
        except Exception as e:
            return False, (None, None, None, None, None)


def get_score(outlined_plan, code, code_return, REWARD_MODEL_LLM, attempts=3, openai_api_key=None):
    e = str()
    for _attempt in range(attempts):
        try:
            # todo: have a reward function here
            sys = (
                f"You are a professor agent who is serving as an expert reward model that can read a research plan, research code, and code output and are able to determine how well a model followed the plan, built the code, and got the proper output scored from 0 to 1 as a float.\n\n"
                f"You must structure your score exactly in the following way: ```SCORE\n<score here>\n``` where SCORE is just the word score, <score here> is a floating point number between 0 and 1 representing how well the model followed the plan, built the code, and got the proper output."
            )
            scoring = query_model(
                model_str=f"{REWARD_MODEL_LLM}",
                system_prompt=sys,
                openai_api_key=openai_api_key,
                prompt=(
                    f"Outlined in the following text is the research plan that the machine learning engineer was tasked with building: {outlined_plan}\n\n"
                    f"The following text is the research code that the model produced: \n{code}\n\n"
                    f"The following is the output from the model: {code_return}\n\n"), temp=0.6)
            performance = extract_prompt(text=scoring, word="SCORE")
            performance = float(performance)
            return performance, f"The performance of your submission is: {performance}", True
        except Exception as e:
            return None, str(e), False
    return 0, e


def code_repair(code, error, ctype, REPAIR_LLM, openai_api_key=None):
    if ctype == "replace":
        repair_sys = (
            "You are an automated code repair tool.\n"
            "Your goal is to take in code and an error and repair the code to make sure the same error does not repeat itself, and also to remove any other potential errors from the code without affecting the code output.\n"
            "Your output should match the original code as closely as possible.\n"
            "You must wrap the code in the following ```python\n<code here>\n```\n"
            "Do not forget the opening ```python and the closing ```."
        )
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"Provided here is the error: {error}\n\nProvided below is the code:\n\n{code}", temp=0.8)
        return extract_prompt(model_resp, "python")
    elif ctype == "edit":
        repair_sys = (
            "You are an automated code repair tool.\n"
            "Your goal is to take in code and an error and repair the code to make sure the same error does not repeat itself, and also to remove any other potential errors from the code without affecting the code output.\n"
            "Your output should match the original code as closely as possible.\n"
            
            "============= CODE EDITING TOOL =============\n"
            "You have access to a code editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with code. \n"
            "You can edit code using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new code that is replacing the old code. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code.\n"
            "Please use the code editing tool to fix this code."
            "Do not forget the opening ```EDIT N M and the closing ```."
            "Your output should look like the following\n\n```EDIT N M\n<new lines to replace old lines>\n```"
        )
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"Provided here is the error: {error}\n\nProvided below is the code:\n\n{code}", temp=0.2)
        return model_resp


class MLESolver:
    def __init__(self, dataset_code, openai_api_key=None, notes=None, max_steps=10, insights=None, plan=None, llm_str=None):
        self.supress_print = False
        if notes is None: self.notes = []
        else: self.notes = notes
        self.dataset_code = dataset_code
        if plan is None: self.plan = ""
        else: self.plan = plan
        self.llm_str = llm_str
        self.verbose = False
        self.max_codes = 1
        self.st_hist_len = 2
        self.min_gen_trials = 1
        self.code_lines = str()
        self.st_history = list()
        self.insights = insights
        self.code_reflect = str()
        self.max_steps = max_steps
        self.prev_code_ret = str()
        self.should_execute_code = True
        self.openai_api_key = openai_api_key

    def initial_solve(self):
        """
        Initialize the solver and get an initial set of code and a return
        @return: None
        """
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@ Initial CodeGen Commands @@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.best_score = None
        self.commands = [Replace()]
        self.model = f"{self.llm_str}"
        init_code, init_return, self.best_score = self.gen_initial_code()
        self.best_codes = [(copy(init_code), self.best_score, init_return) for _ in range(1)]

        self.code_lines = init_code
        self.model = f"{self.llm_str}"
        self.commands = [Edit(), Replace()]
        self.prev_working_code = copy(self.code_lines)

    @staticmethod
    def clean_text(text):
        text = text.replace("```\n", "```")
        text = text.replace("```python\n", "```REPLACE\n")
        return text

    def gen_initial_code(self):
        num_attempts = 0
        error_hist = list()
        while True:
            if num_attempts == 0:
                err = str()
                err_hist = str()
            else:
                err = f"The following was the previous command generated: {model_resp}. This was the error return {cmd_str}. You should make sure not to repeat this error and to solve the presented problem."
                error_hist.append(err)
                if len(error_hist) == 5: _ = error_hist.pop(0)
                err = "\n".join(error_hist)
                err_hist = "The following is a history of your previous errors\n" + err + "\nDO NOT REPEAT THESE."
            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"{err_hist}\nYou should now use ```REPLACE to create initial code to solve the challenge. Now please enter the ```REPLACE command below:\n ", temp=1.0)
            model_resp = self.clean_text(model_resp)
            cmd_str, code_lines, prev_code_ret, should_execute_code, score = self.process_command(model_resp)
            if not self.supress_print: print(f"@@@ INIT ATTEMPT: Command Exec // Attempt {num_attempts}: ", str(cmd_str).replace("\n", " | "))
            if not self.supress_print: print(f"$$$ Score: {score}")
            if score is not None: break
            num_attempts += 1
        return code_lines, prev_code_ret, score

    def solve(self):
        num_attempts = 0
        best_pkg = None
        top_score = None
        self.prev_code_ret = None
        self.should_execute_code = False
        while True:
            if len(self.commands) == 2: cmd_app_str = "You must output either the ```EDIT or ```REPLACE command immediately. "
            else: cmd_app_str = ""
            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"The following is your history:{self.history_str()}\n\n{cmd_app_str}Now please enter a command: ", temp=1.0)
            model_resp = self.clean_text(model_resp)
            self.code_lines = copy(random.choice(self.best_codes)[0])
            cmd_str, code_lines, prev_code_ret, should_execute_code, score = self.process_command(model_resp)
            self.st_history.append([model_resp, prev_code_ret, code_lines, cmd_str])
            if len(self.st_history) > self.st_hist_len: self.st_history.pop(0)
            if score is not None:
                if top_score is None:
                    best_pkg = copy(code_lines), copy(prev_code_ret), copy(should_execute_code), copy(model_resp), copy(cmd_str)
                    top_score = score
                elif score > top_score:
                    best_pkg = copy(code_lines), copy(prev_code_ret), copy(should_execute_code), copy(model_resp), copy(cmd_str)
                    top_score = score
            if not self.supress_print: print(f"@@@ Command Exec // Attempt {num_attempts}: ", str(cmd_str).replace("\n", " | "))
            if not self.supress_print: print(f"$$$ Score: {score}")
            if num_attempts >= self.min_gen_trials and top_score is not None: break
            num_attempts += 1
        self.code_lines, self.prev_code_ret, self.should_execute_code, model_resp, cmd_str = best_pkg
        if not self.supress_print: print(prev_code_ret)
        # add top scoring code that was successful to the best codes
        if top_score > self.best_codes[-1][1]:
            # replace the lowest scoring one
            if len(self.best_codes) >= self.max_codes:
                self.best_codes.pop(-1)
                self.code_reflect = self.reflect_code()
            self.best_codes.append((copy(self.code_lines), copy(top_score), self.prev_code_ret))
            # sort by score, to make sure lowest are removed in future
            self.best_codes.sort(key=lambda x: x[1], reverse=True)
        return model_resp, cmd_str

    def reflect_code(self):
        """
        Provide a reflection on produced behavior for next execution
        @return: (str) language model-produced reflection
        """
        code_strs = ("$"*40 + "\n\n").join([self.generate_code_lines(_code[0]) + f"\nCode Return {_code[1]}" for _code in self.best_codes])
        code_strs = f"Please reflect on the following sets of code: {code_strs} and come up with generalizable insights that will help you improve your performance on this benchmark."
        syst = self.system_prompt(commands=False) + code_strs
        return query_model(prompt="Please reflect on ideas for how to improve your current code. Examine the provided code and think very specifically (with precise ideas) on how to improve performance, which methods to use, how to improve generalization on the test set with line-by-line examples below:\n", system_prompt=syst, model_str=f"{self.llm_str}", openai_api_key=self.openai_api_key)

    def process_command(self, model_resp):
        """
        Take command from language model and execute if valid
        @param model_resp: (str) language model output
        @return: (tuple) tuple containing the following items
            - cmd_str: (str) code execution return and success flag
            - code_lines: (list) list of code lines as strings
            - prev_code_ret: (str) output from running code
            - should_execute_code: (bool) did the code change, if so we need to re-execute it
            - score: (float) score of model
        """
        prev_code_ret = self.prev_code_ret
        should_execute_code = self.should_execute_code
        code_lines = copy(self.code_lines)
        remove_figures()
        for cmd in self.commands:
            if cmd.matches_command(model_resp):
                # attempt to execute the code edit command
                if cmd.cmd_type == "CODE-edit":
                    score = None
                    failed = True
                    code_err = str()
                    for _tries in range(GLOBAL_REPAIR_ATTEMPTS):
                        success, args = cmd.parse_command(model_resp, copy(self.code_lines), self.dataset_code)
                        if success:
                            cmd_return = cmd.execute_command(args)
                            code_err = f"Return from executing code: {cmd_return[2]}"
                            if cmd_return[0]:  # if success
                                code_lines = copy(cmd_return[1])
                                score, cmd_str, is_valid = get_score(self.plan, "\n".join(code_lines), cmd_return[2], openai_api_key=self.openai_api_key, REWARD_MODEL_LLM=self.llm_str)
                                if is_valid:
                                    failed = False
                                    break
                                code_err += f"\nReturn from executing code on real test set {cmd_str}"
                        repaired_code = code_repair(model_resp, code_err, REPAIR_LLM=self.llm_str, ctype="edit", openai_api_key=self.openai_api_key)
                        model_resp = repaired_code
                        if not self.supress_print: print(f"     * Attempting repair // try {_tries}*")
                    if failed:
                        cmd_str = f"Code editing FAILED due to the following error: {code_err}. Code was reverted back to original state before edits."
                        if not self.supress_print: print("$$$$ CODE EDIT (failed)")
                    else:
                        cmd_str = "Code was successfully edited."
                        prev_code_ret = copy(cmd_return[2])
                        if not self.supress_print: print("$$$$ CODE EDIT (success)")
                        should_execute_code = True
                    return cmd_str, code_lines, prev_code_ret, should_execute_code, score
                # attempt to execute the code replace command
                elif cmd.cmd_type == "CODE-replace": # DONE
                    score = None
                    failed = True
                    code_err = str()
                    for _tries in range(GLOBAL_REPAIR_ATTEMPTS):
                        success, args = cmd.parse_command(model_resp, self.dataset_code)
                        code_err = f"Return from executing code: {args[1]}"
                        if success:
                            code_lines = copy(args[0])
                            score, cmd_str, is_valid = get_score(self.plan, "\n".join(code_lines), args[1], openai_api_key=self.openai_api_key, REWARD_MODEL_LLM=self.llm_str)
                            if is_valid:
                                failed = False
                                break
                            code_err += f"\nReturn from executing code on real test set {cmd_str}"
                        repaired_code = code_repair(extract_prompt(model_resp, "REPLACE", ), code_err, ctype="replace", openai_api_key=self.openai_api_key, REPAIR_LLM=self.llm_str)
                        repaired_code = f"```REPLACE\n{repaired_code}\n```"
                        model_resp = repaired_code
                        if not self.supress_print: print(f"     * Attempting repair // try {_tries}*")
                    if failed:
                        cmd_str = f"Code replacement FAILED due to the following error: {code_err}.  Code was reverted back to original state before edits."
                        if not self.supress_print: print("$$$$ CODE REPLACE (failed)")
                    else:
                        cmd_str = "Code was successfully replaced."
                        code_lines = copy(args[0])
                        prev_code_ret = copy(args[1])
                        if not self.supress_print: print("$$$$ CODE REPLACE (success)")
                        should_execute_code = True
                    return cmd_str, code_lines, prev_code_ret, should_execute_code, score
        if not self.supress_print: print("$$$$ INVALID COMMAND (failed)")
        return "Command not supported, choose from existing commands", None, None, None, None

    def history_str(self):
        """
        Well-formatted history string
        @return: (str) history string
        """
        hist_str = ""
        for _hist in range(len(self.st_history)):
            hist_str += f"-------- History ({len(self.st_history)-_hist} steps ago) -----\n"
            hist_str += f"Because of the following response: {self.st_history[_hist][0]}\n" if len(self.st_history[_hist][0]) > 0 else ""
            hist_str += f"and the following COMMAND response output: {self.st_history[_hist][3]}\n"
            hist_str += f"With the following code used: {'#'*20}\n{self.st_history[_hist][2]}\n{'#'*20}\n\n"
            hist_str += f"The environment feedback and reflection was as follows: {self.st_history[_hist][1]}\n"
            hist_str += f"-------- End of history ({len(self.st_history)-_hist} steps ago) -------\n"
        return hist_str

    def system_prompt(self, commands=True):
        """
        Produce a system prompt for the mle-solver to solve ml problems
        @param commands: (bool) whether to use command prompt
        @return: (str) system prompt
        """
        return (
            # ROLE DESCRIPTION
            f"{self.role_description()}.\n"
            # TASK INSTRUCTIONS
            f"The following are your task instructions: {self.phase_prompt()}\n"
            # LIT REVIEW INSIGHTS
            f"Provided below are some insights from a literature review summary:\n{self.insights}\n"
            # CODE INSIGHTS
            f"{self.code_reflect}"
            # NOTES
            f"The following are notes, instructions, and general tips for you: {self.notes}"
            # PLAN DESCRIPTION
            f"You are given a machine learning research task described, where the plan is described as follows: {self.plan}\n"
            # DATASET DESCRIPTION            
            f"{self.generate_dataset_descr_prompt()}"
            # Create Figures
            f"You should also try generating at least two figures to showcase the results, titled Figure_1.png and Figure_2.png\n"
            f"Your method MUST not get 0% accuracy. If it does, you have done something wrong and must correct this. Make sure to check your accuracy calculation is correct.\n"
            # transition
            f"Your goal is to solve the research plan as well as possible. You will receive a score after you write the code and should aim to maximize the score by following the plan instructions and writing high quality code.\n"
            f"Before each experiment please include a print statement explaining exactly what the results are meant to show in great detail before printing the results out.\n"
            # COMMAND SET
            f"The following are commands you have access to: {self.command_descriptions()}\n. You should try to have a diversity of command responses if appropriate. Do not repeat the same commend too many times. Please consider looking through your history and not repeating commands too many times." if commands else ""
        )

    def generate_code_lines(self, code):
        """
        Generate well-formatted code lines with line numbers
        @param code: (list) list of code line strings
        @return: (str) code lines formatted with line numbers
        """
        codestr = str()
        for _index in range(len(code)):
            codestr += f"{_index} |{code[_index]}\n"
        return codestr

    def feedback(self, code_return):
        """
        Provide execution feedback after command is run
        @param code_return: (str) return from code execution
        @return: (str) feedback string
        """
        if code_return is not None:
            code_str = self.generate_code_lines(self.code_lines)
            if "[CODE EXECUTION ERROR]" in code_return:
                if not self.supress_print: print(f"@@@@ ERROR")  # , {code_return.replace('\n', '')}")
                reflect_prompt = f"This is your code: {code_str}\n\nYour code returned the following error {code_return}. Please provide a detailed reflection on why this error was returned, which lines in the code caused this error, and exactly (line by line) how you hope to fix this in the next update. This step is mostly meant to reflect in order to help your future self fix the error better. Do not provide entirely new code but provide suggestions on how to fix the bug using LINE EDITS."
            elif os.path.exists("submission.csv"):
                self.prev_working_code = copy(self.code_lines)
                grade_return = get_score(self.plan, "\n".join(self.prev_working_code), code_return, openai_api_key=self.openai_api_key)[0]
                if not self.supress_print: print(f"@@@@ SUBMISSION: model score {grade_return}", REWARD_MODEL_LLM=self.llm_str)
                f"Your code was properly submitted and you have just received a grade for your model.\nYour score was {grade_return}.\n\n"
                reflect_prompt = f"This is your code: {code_str}\n\nYour code successfully returned a submission csv. Consider further improving your technique through advanced learning techniques, data augmentation, or hyperparamter tuning to increase the score. Please provide a detailed reflection on how to improve your performance, which lines in the code could be improved upon, and exactly (line by line) how you hope to improve this in the next update. This step is mostly meant to reflect in order to help your future self."

                for file in os.listdir("."):
                    if file.endswith(".csv"):
                        os.system(f"rm {file}")
            else:
                if not self.supress_print: print("@@@@ No return")
                reflect_prompt = f"This is your code: {code_str}\n\nYour code did not return an error, but also did not successfully submit a submission csv file. Please reflect on how you can improve your submission for the next cycle to submit a file and obtain a high score."
        elif not self.should_execute_code:
            code_return = "No changes were made to the code."
            reflect_prompt = "Reflect on your future plans and next steps to improve the code."
        reflection = self.reflection(reflect_prompt, code_str, code_return)
        return f"Code return: {code_return}\n\nReflection: {reflection}"

    def reflection(self, reflect_prompt, code_str, code_return):
        """
        Reflect on your future plans and next steps to improve the code
        @param reflect_prompt: (str) reflection prompt
        @param code_str: (str) code string
        @return: (str) reflection string
        """
        refl = query_model(prompt=reflect_prompt, system_prompt=self.system_prompt(commands=False), model_str=f"{self.llm_str}", openai_api_key=self.openai_api_key)
        return f"During the previous execution, the following code was run: \n\n{code_str}\n\nThis code returned the following: \n{code_return}\nThe following is your reflection from this feedback {refl}\n"

    def generate_dataset_descr_prompt(self):
        """
        Generate description prompt for kaggle dataset
        @param data_loader: (DataLoader) data loader
        @return: (str) description prompt
        """
        return f"\n- The following dataset code will be added to the beginning of your code always, so this does not need to be rewritten: {self.dataset_code}"

    def phase_prompt(self,):
        """
        Describe system role and general tips for mle-solver
        @return: (str) system role
        """
        phase_str = (
            "You are an ML engineer and you will be writing the code for a research project.\n"
            "Your goal is to produce code that obtains final results for a set of research experiments. You should aim for simple code to collect all results, not complex code. You should integrate the provided literature review and the plan to make sure you are implementing everything outlined in the plan. The dataset code will be added to the beginning of your code always, so this does not need to be rewritten. Make sure you do not write functions, only loose code.\n"
            "I would recommend writing smaller code so you do not run out of time but make sure to work on all points in the plan in the same code. You code should run every experiment outlined in the plan for a single code.\n"
            "You cannot pip install new libraries, but many machine learning libraries already work. If you wish to use a language model in your code, please use the following:\nAnything you decide to print inside your code will be provided to you as input, and you will be able to see that part of the code. Using print statements is useful for figuring out what is wrong and understanding your code better."
        )
        return phase_str

    def role_description(self):
        """
        Provide role description
        @return: (str) role description
        """
        return "You are an expert machine learning engineer working at a top university to write code to solve machine learning research challenges using your machine learning expertise."

    @staticmethod
    def _common_code_errors():
        """
        Some general tips to avoid common code errors, also TF has many errors so we avoid this and ask to use pytorch
        @return: (str) common code errors
        """
        return (
            "Make sure to import everything that you are using.\n"
            "Reflect on the code before writing it to make sure there are no bugs or compilation issues.\n"
            "YOU MUST USE COMMANDS PROPERLY. Do not use the word COMMAND for the command that is incorrect. You must use an actual command (e.g. EDIT, REPLACE...) NOT THE WORD COMMAND. Do not make this mistake.\n"
            "Under no circumstances should you use tensorflow or keras. Only use pytorch for scikitlearn for deep learning.\n"
        )

    def command_descriptions(self):
        """
        Provide command descriptions
        @return: (str) command descriptions
        """
        cmd_strings = "\n".join([_cmd.docstring() for _cmd in self.commands])
        return f"\nYou also have access to tools which can be interacted with using the following structure: ```COMMAND\n<command information here>\n```, where COMMAND is whichever command you want to run (e.g. EDIT, REPLACE...), <command information here> is information used for the command, such as code to run or a search query, and ``` are meant to encapsulate the command. ``` must be included as part of the command both at the beginning and at the end of the code. DO NOT FORGOT TO HAVE ``` AT THE TOP AND BOTTOM OF CODE. and this structure must be followed to execute a command correctly. YOU CAN ONLY EXECUTE A SINGLE COMMAND AT A TIME! Do not try to perform multiple commands EVER only one. {self._common_code_errors()}" + cmd_strings

    def run_code(self):
        """
        Actually execute the code that was generated
        @return: (str) code return
        """
        if self.prev_code_ret is not None:
            return self.prev_code_ret
        elif self.should_execute_code:
            return execute_code("\n".join(self.code_lines))
        return "Changes have not yet been made to the code."




