# ai_lab_repo.py
import threading
import argparse
import pickle
import yaml
import requests  # explicit import for network calls
from pathlib import Path
from datetime import date

from pypdf import PdfReader  # unify on pypdf
from app import *
from agents import *
from copy import copy
from common_imports import *
from mlesolver import MLESolver

GLOBAL_AGENTRXIV = None
DEFAULT_LLM_BACKBONE = "o3-mini"
RESEARCH_DIR_PATH = "MATH_research_dir"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LaboratoryWorkflow:
    def __init__(
        self,
        research_topic,
        openai_api_key,
        max_steps=100,
        num_papers_lit_review=5,
        agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}",
        notes=list(),
        human_in_loop_flag=None,
        compile_pdf=True,
        mlesolver_max_steps=3,
        papersolver_max_steps=5,
        paper_index=0,
        except_if_fail=False,
        parallelized=False,
        lab_dir=None,
        lab_index=0,
        agentRxiv=False,
        agentrxiv_papers=5
    ):
        """
        Initialize laboratory workflow
        """
        self.agentRxiv = agentRxiv
        self.max_prev_papers = 10
        self.parallelized = parallelized
        self.notes = notes
        self.lab_dir = lab_dir
        self.lab_index = lab_index
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.paper_index = paper_index
        self.openai_api_key = openai_api_key
        self.except_if_fail = except_if_fail
        self.research_topic = research_topic
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review

        self.print_cost = True
        self.review_override = True  # should review be overridden?
        self.review_ovrd_steps = 0   # review steps so far
        self.arxiv_paper_exp_time = 3
        self.reference_papers = list()

        ##########################################
        ####### COMPUTE BUDGET PARAMETERS ########
        ##########################################
        self.num_ref_papers = 1
        self.review_total_steps = 0
        self.arxiv_num_summaries = 5
        self.num_agentrxiv_papers = agentrxiv_papers
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps

        self.phases = [
            ("literature review", ["literature review"]),
            ("plan formulation", ["plan formulation"]),
            ("experimentation", ["data preparation", "running experiments"]),
            ("results interpretation", ["results interpretation", "report writing", "report refinement"]),
        ]
        self.phase_status = {subtask: False for _, subtasks in self.phases for subtask in subtasks}

        self.phase_models = dict()
        if isinstance(agent_model_backbone, str):
            for _, subtasks in self.phases:
                for subtask in subtasks:
                    self.phase_models[subtask] = agent_model_backbone
        elif isinstance(agent_model_backbone, dict):
            self.phase_models = agent_model_backbone

        self.human_in_loop_flag = human_in_loop_flag

        self.statistics_per_phase = {
            "literature review":      {"time": 0.0, "steps": 0.0},
            "plan formulation":       {"time": 0.0, "steps": 0.0},
            "data preparation":       {"time": 0.0, "steps": 0.0},
            "running experiments":    {"time": 0.0, "steps": 0.0},
            "results interpretation": {"time": 0.0, "steps": 0.0},
            "report writing":         {"time": 0.0, "steps": 0.0},
            "report refinement":      {"time": 0.0, "steps": 0.0},
        }

        self.save = True
        self.verbose = True
        self.reviewers = ReviewersAgent(model=self.model_backbone, notes=self.notes, openai_api_key=self.openai_api_key)
        self.phd = PhDStudentAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.postdoc = PostdocAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.professor = ProfessorAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.ml_engineer = MLEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.sw_engineer = SWEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)

    def set_model(self, model):
        self.set_agent_attr("model", model)
        self.reviewers.model = model

    def save_state(self, phase):
        with open(f"state_saves/Paper{self.paper_index}.pkl", "wb") as f:
            pickle.dump(self, f)

    def set_agent_attr(self, attr, obj):
        setattr(self.phd, attr, obj)
        setattr(self.postdoc, attr, obj)
        setattr(self.professor, attr, obj)
        setattr(self.ml_engineer, attr, obj)
        setattr(self.sw_engineer, attr, obj)

    def reset_agents(self):
        self.phd.reset()
        self.postdoc.reset()
        self.professor.reset()
        self.ml_engineer.reset()
        self.sw_engineer.reset()

    def perform_research(self):
        for phase, subtasks in self.phases:
            phase_start_time = time.time()
            if self.verbose:
                hdr = f"[Lab #{self.lab_index} Paper #{self.paper_index}] " if self.agentRxiv else ""
                print(f"{'*' * 50}\nBeginning phase: {hdr}{phase}\n{'*' * 50}")
            for subtask in subtasks:
                if isinstance(self.phase_models, dict):
                    self.set_model(self.phase_models.get(subtask, f"{DEFAULT_LLM_BACKBONE}"))
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "literature review":
                    repeat = True
                    while repeat:
                        repeat = self.literature_review()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "plan formulation":
                    repeat = True
                    while repeat:
                        repeat = self.plan_formulation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "data preparation":
                    repeat = True
                    while repeat:
                        repeat = self.data_preparation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "running experiments":
                    repeat = True
                    while repeat:
                        repeat = self.running_experiments()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "results interpretation":
                    repeat = True
                    while repeat:
                        repeat = self.results_interpretation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report writing":
                    repeat = True
                    while repeat:
                        repeat = self.report_writing()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report refinement":
                    return_to_exp_phase = self.report_refinement()
                    if not return_to_exp_phase:
                        if self.save:
                            self.save_state(subtask)
                        return
                    self.set_agent_attr("second_round", return_to_exp_phase)
                    self.set_agent_attr("prev_report", copy(self.phd.report))
                    self.set_agent_attr("prev_exp_results", copy(self.phd.exp_results))
                    self.set_agent_attr("prev_results_code", copy(self.phd.results_code))
                    self.set_agent_attr("prev_interpretation", copy(self.phd.interpretation))
                    self.phase_status["plan formulation"] = False
                    self.phase_status["data preparation"] = False
                    self.phase_status["running experiments"] = False
                    self.phase_status["results interpretation"] = False
                    self.phase_status["report writing"] = False
                    self.phase_status["report refinement"] = False
                    self.perform_research()
                if self.save:
                    self.save_state(subtask)
                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time
                print(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                self.statistics_per_phase[subtask]["time"] = phase_duration

    def report_refinement(self):
        reviews = self.reviewers.inference(self.phd.plan, self.phd.report)
        print("Reviews:", reviews)
        if self.human_in_loop_flag["report refinement"]:
            print(f"Provided are reviews from a set of three reviewers: {reviews}")
            input("Would you like to be completed with the project or should the agents go back and improve their experimental results?\n (y) for go back (n) for complete project: ")
        else:
            review_prompt = (
                f"Provided are reviews from a set of three reviewers: {reviews}. Would you like to be completed with the project or do you want to go back to the planning phase and improve your experiments?\n "
                f"Type y and nothing else to go back, type n and nothing else for complete project."
            )
            self.phd.phases.append("report refinement")
            if self.review_override:
                if self.review_total_steps == self.review_ovrd_steps:
                    response = "n"
                else:
                    response = "y"
                    self.review_ovrd_steps += 1
            else:
                response = self.phd.inference(
                    research_topic=self.research_topic, phase="report refinement", feedback=review_prompt, step=0)
            if len(response) == 0:
                raise Exception("Model did not respond")
            response = response.lower().strip()[0]
            if response == "n":
                if self.verbose:
                    print("*" * 40, "\n", "REVIEW COMPLETE", "\n", "*" * 40)
                return False
            elif response == "y":
                self.set_agent_attr("reviewer_response", f"Provided are reviews from a set of three reviewers: {reviews}.")
                return True
            else:
                raise Exception("Model did not respond")

    def report_writing(self):
        """
        Perform report writing phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        report_notes = [_note["note"] for _note in self.ml_engineer.notes if "report writing" in _note["phases"]]
        report_notes = f"Notes for the task objective: {report_notes}\n" if len(report_notes) > 0 else ""
        # instantiate paper-solver
        from papersolver import PaperSolver
        self.reference_papers = []
        # Pick model safely: dict per-phase or fallback to string backbone
        report_model = self.phase_models["report writing"] if isinstance(self.phase_models, dict) else self.model_backbone
        solver = PaperSolver(
            notes=report_notes,
            max_steps=self.papersolver_max_steps,
            plan=self.phd.plan,
            exp_code=self.phd.results_code,
            exp_results=self.phd.exp_results,
            insights=self.phd.interpretation,
            lit_review=self.phd.lit_review,
            ref_papers=self.reference_papers,
            topic=self.research_topic,
            openai_api_key=self.openai_api_key,
            llm_str=report_model,
            compile_pdf=self.compile_pdf,
            save_loc=self.lab_dir
        )
        # run initialization for solver
        solver.initial_solve()
        # run solver steps
        for _ in range(self.papersolver_max_steps):
            solver.solve()
        # get best report results
        report = "\n".join(solver.best_report[0][0])
        score = solver.best_report[0][1]
        match = re.search(r"\\title\{([^}]*)\}", report)
        if match:
            report_title = match.group(1).replace(" ", "_")
        else:
            report_title = "_".join([str(random.randint(0, 10)) for _ in range(10)])
        if self.agentRxiv:
            shutil.copyfile(os.path.join(self.lab_dir, "tex", "temp.pdf"), f"uploads/{report_title}.pdf")
        if self.verbose:
            print(f"Report writing completed, reward function score: {score}")
        if self.human_in_loop_flag["report writing"]:
            retry = self.human_in_loop("report writing", report)
            if retry:
                return retry
        self.set_agent_attr("report", report)
        readme = self.professor.generate_readme()
        save_to_file(f"./{self.lab_dir}", "readme.md", readme)
        save_to_file(f"./{self.lab_dir}", "report.txt", report)
        self.reset_agents()
        return False

    def results_interpretation(self):
        max_tries = self.max_steps
        dialogue = str()
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            resp = self.postdoc.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose:
                print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose:
                    print("#" * 40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#" * 40)
            if "```INTERPRETATION" in resp:
                interpretation = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag["results interpretation"]:
                    retry = self.human_in_loop("results interpretation", interpretation)
                    if retry:
                        return retry
                self.set_agent_attr("interpretation", interpretation)
                self.reset_agents()
                self.statistics_per_phase["results interpretation"]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose:
                print("PhD Student: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose:
                    print("#" * 40, "\n", "PhD Dialogue:", dialogue, "#" * 40, "\n")
        raise Exception("Max tries during phase: Results Interpretation")

    def running_experiments(self):
        experiment_notes = [_note["note"] for _note in self.ml_engineer.notes if "running experiments" in _note["phases"]]
        experiment_notes = f"Notes for the task objective: {experiment_notes}\n" if len(experiment_notes) > 0 else ""
        solver = MLESolver(
            dataset_code=self.ml_engineer.dataset_code,
            notes=experiment_notes,
            insights=self.ml_engineer.lit_review_sum,
            max_steps=self.mlesolver_max_steps,
            plan=self.ml_engineer.plan,
            openai_api_key=self.openai_api_key,
            llm_str=self.model_backbone if isinstance(self.phase_models, str) else self.phase_models["running experiments"],
        )
        solver.initial_solve()
        for _ in range(self.mlesolver_max_steps - 1):
            solver.solve()
        code = "\n".join(solver.best_codes[0][0])
        score = solver.best_codes[0][1]
        exp_results = solver.best_codes[0][2]
        if self.verbose:
            print(f"Running experiments completed, reward function score: {score}")
        if self.human_in_loop_flag["running experiments"]:
            retry = self.human_in_loop("data preparation", code)
            if retry:
                return retry
        save_to_file(f"./{self.lab_dir}/src", "run_experiments.py", code)
        save_to_file(f"./{self.lab_dir}/src", "experiment_output.log", exp_results)
        self.set_agent_attr("results_code", code)
        self.set_agent_attr("exp_results", exp_results)
        self.reset_agents()
        return False

    def data_preparation(self):
        max_tries = self.max_steps
        ml_feedback = str()
        ml_dialogue = str()
        swe_feedback = str()
        ml_command = str()
        hf_engine = HFDataSearch()
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            ml_feedback_in = f"Feedback provided to the ML agent: {ml_feedback}" if ml_feedback != "" else ""
            resp = self.sw_engineer.inference(
                self.research_topic, "data preparation",
                feedback=f"{ml_dialogue}\nFeedback from previous command: {swe_feedback}\n{ml_command}{ml_feedback_in}", step=_i)
            swe_feedback = str()
            swe_dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                swe_dialogue = f"\nThe following is dialogue produced by the SW Engineer: {dialogue}\n"
                if self.verbose:
                    print("#" * 40, f"\nThe following is dialogue produced by the SW Engineer: {dialogue}", "\n", "#" * 40)
            if "```SUBMIT_CODE" in resp:
                final_code = extract_prompt(resp, "SUBMIT_CODE")
                code_resp = execute_code(final_code, timeout=60)
                if self.verbose:
                    print("!" * 100, "\n", f"CODE RESPONSE: {code_resp}")
                swe_feedback += f"\nCode Response: {code_resp}\n"
                if "[CODE EXECUTION ERROR]" in code_resp:
                    swe_feedback += "\nERROR: Final code had an error and could not be submitted! You must address and fix this error.\n"
                else:
                    if self.human_in_loop_flag["data preparation"]:
                        retry = self.human_in_loop("data preparation", final_code)
                        if retry:
                            return retry
                    save_to_file(f"./{self.lab_dir}/src", "load_data.py", final_code)
                    self.set_agent_attr("dataset_code", final_code)
                    self.reset_agents()
                    self.statistics_per_phase["data preparation"]["steps"] = _i
                    return False

            ml_feedback_in = f"Feedback from previous command: {ml_feedback}" if ml_feedback != "" else ""
            resp = self.ml_engineer.inference(
                self.research_topic, "data preparation",
                feedback=f"{swe_dialogue}\n{ml_feedback_in}", step=_i)
            ml_feedback = str()
            ml_dialogue = str()
            ml_command = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                ml_dialogue = f"\nThe following is dialogue produced by the ML Engineer: {dialogue}\n"
                if self.verbose:
                    print("#" * 40, f"\nThe following is dialogue produced by the ML Engineer: {dialogue}", "#" * 40, "\n")
            if "```python" in resp:
                code = extract_prompt(resp, "python")
                code = self.ml_engineer.dataset_code + "\n" + code
                code_resp = execute_code(code, timeout=120)
                ml_command = f"Code produced by the ML agent:\n{code}"
                ml_feedback += f"\nCode Response: {code_resp}\n"
                if self.verbose:
                    print("!" * 100, "\n", f"CODE RESPONSE: {code_resp}")
            if "```SEARCH_HF" in resp:
                hf_query = extract_prompt(resp, "SEARCH_HF")
                hf_res = "\n".join(hf_engine.results_str(hf_engine.retrieve_ds(hf_query)))
                ml_command = f"HF search command produced by the ML agent:\n{hf_query}"
                ml_feedback += f"Huggingface results: {hf_res}\n"
        raise Exception("Max tries during phase: Data Preparation")

    def plan_formulation(self):
        max_tries = self.max_steps
        dialogue = str()
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            resp = self.postdoc.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose:
                print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose:
                    print("#" * 40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#" * 40)

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                if self.human_in_loop_flag["plan formulation"]:
                    retry = self.human_in_loop("plan formulation", plan)
                    if retry:
                        return retry
                self.set_agent_attr("plan", plan)
                self.reset_agents()
                self.statistics_per_phase["plan formulation"]["steps"] = _i
                return False

            resp = self.phd.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose:
                print("PhD Student: ", resp, "\n~~~~~~~~~~~")

            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose:
                    print("#" * 40, "\n", "PhD Dialogue:", dialogue, "#" * 40, "\n")
        if self.except_if_fail:
            raise Exception("Max tries during phase: Plan Formulation")
        else:
            plan = "No plan specified."
            if self.human_in_loop_flag["plan formulation"]:
                retry = self.human_in_loop("plan formulation", plan)
                if retry:
                    return retry
            self.set_agent_attr("plan", plan)
            self.reset_agents()
            return False

    def literature_review(self):
        arx_eng = ArxivSearch()
        max_tries = self.max_steps
        resp = self.phd.inference(self.research_topic, "literature review", step=0, temp=0.4)
        if self.verbose:
            print(resp, "\n~~~~~~~~~~~")
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            feedback = str()
            if "```SUMMARY" in resp:
                query = extract_prompt(resp, "SUMMARY")
                papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                if self.agentRxiv and GLOBAL_AGENTRXIV.num_papers() > 0:
                    papers += GLOBAL_AGENTRXIV.search_agentrxiv(query, self.num_agentrxiv_papers)
                feedback = f"You requested arXiv papers related to the query {query}, here was the response\n{papers}"

            elif "```FULL_TEXT" in resp:
                query = extract_prompt(resp, "FULL_TEXT")
                if self.agentRxiv and "AgentRxiv" in query:
                    full_text = GLOBAL_AGENTRXIV.retrieve_full_text(query,)
                else:
                    full_text = arx_eng.retrieve_full_paper_text(query)
                arxiv_paper = f"```EXPIRATION {self.arxiv_paper_exp_time}\n" + full_text + "```"
                feedback = arxiv_paper

            elif "```ADD_PAPER" in resp:
                query = extract_prompt(resp, "ADD_PAPER")
                if self.agentRxiv and "AgentRxiv" in query:
                    feedback, text = self.phd.add_review(query, arx_eng, agentrxiv=True, GLOBAL_AGENTRXIV=GLOBAL_AGENTRXIV)
                else:
                    feedback, text = self.phd.add_review(query, arx_eng)
                if len(self.reference_papers) < self.num_ref_papers:
                    self.reference_papers.append(text)

            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                lit_review_sum = self.phd.format_review()
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    if retry:
                        self.phd.lit_review = []
                        return retry
                if self.verbose:
                    print(self.phd.lit_review_sum)
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, "literature review", feedback=feedback, step=_i + 1, temp=0.4)
            if self.verbose:
                print(resp, "\n~~~~~~~~~~~")
        if self.except_if_fail:
            raise Exception("Max tries during phase: Literature Review")
        else:
            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                lit_review_sum = self.phd.format_review()
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    if retry:
                        self.phd.lit_review = []
                        return retry
                if self.verbose:
                    print(self.phd.lit_review_sum)
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False

    def human_in_loop(self, phase, phase_prod):
        print("\n" * 5)
        print(f"Presented is the result of the phase [{phase}]: {phase_prod}")
        y_or_no = None
        while y_or_no not in ["y", "n"]:
            y_or_no = input("\n\n\nAre you happy with the presented content? Respond Y or N: ").strip().lower()
            if y_or_no == "y":
                pass
            elif y_or_no == "n":
                notes_for_agent = input("Please provide notes for the agent so that they can try again and improve performance: ")
                self.reset_agents()
                self.notes.append({"phases": [phase], "note": notes_for_agent})
                return True
            else:
                print("Invalid response, type Y or N")
        return False


class AgentRxiv:
    def __init__(self, lab_index=0):
        self.lab_index = lab_index
        self.server_thread = None
        self.initialize_server()
        self.pdf_text = dict()
        self.summaries = dict()

    def initialize_server(self):
        port = 5000 + self.lab_index
        self.server_thread = threading.Thread(target=lambda: self.run_server(port))
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(5)  # allow time for the server to start up

    @staticmethod
    def num_papers():
        return len(os.listdir("uploads"))

    def retrieve_full_text(self, arxiv_id):
        try:
            return self.pdf_text[arxiv_id]
        except Exception:
            return "Paper ID not found?"

    @staticmethod
    def read_pdf_pypdf(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def search_agentrxiv(self, search_query, num_papers):
        url = f"http://127.0.0.1:{5000 + self.lab_index}/api/search?q={search_query}"
        return_str = str()
        try:
            with app.app_context():
                update_papers_from_uploads()
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return_str += "Search Query:" + data["query"]
            return_str += "Results:"
            for result in data["results"][:num_papers]:
                arxiv_id = f"AgentRxiv:ID_{result['id']}"
                if arxiv_id not in self.summaries:
                    filename = Path(f"_tmp_{self.lab_index}.pdf")
                    resp2 = requests.get(result["pdf_url"], timeout=10)
                    filename.write_bytes(resp2.content)
                    self.pdf_text[arxiv_id] = self.read_pdf_pypdf(f"_tmp_{self.lab_index}.pdf")
                    self.summaries[arxiv_id] = query_model(
                        prompt=self.pdf_text[arxiv_id],
                        system_prompt="Please provide a 5 sentence summary of this paper.",
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        model_str="gpt-4o-mini"
                    )
                return_str += f"Title: {result['filename']}"
                return_str += f"Summary: {self.summaries[arxiv_id]}\n"
                formatted_date = date.today().strftime("%d/%m/%Y")
                return_str += f"Publication Date: {formatted_date}\n"
                return_str += f"arXiv paper ID: AgentRxiv:ID_{result['id']}"
                return_str += "-" * 40
        except Exception as e:
            print(f"AgentRxiv Error: {e}")
            return_str += f"Error: {e}"
        return return_str

    def run_server(self, port):
        run_app(port=port)


def parse_arguments():
    parser = argparse.ArgumentParser(description="AgentLaboratory Research Workflow")
    parser.add_argument(
        "--yaml-location",
        type=str,
        default="experiment_configs/MATH_agentlab.yaml",
        help="Location of YAML to load config data."
    )
    return parser.parse_args()


def parse_yaml(yaml_file_loc):
    with open(yaml_file_loc, "r") as file:
        agentlab_data = yaml.safe_load(file)

    class YamlDataHolder:
        def __init__(self):
            pass

    parser = YamlDataHolder()

    # Accept both canonical keys and synonyms from older configs
    # copilot_mode / copilot-mode
    if "copilot_mode" in agentlab_data:
        parser.copilot_mode = agentlab_data["copilot_mode"]
    elif "copilot-mode" in agentlab_data:
        parser.copilot_mode = agentlab_data["copilot-mode"]
    else:
        parser.copilot_mode = False

    # load-previous / load-existing
    if "load-previous" in agentlab_data:
        parser.load_previous = agentlab_data["load-previous"]
    elif "load-existing" in agentlab_data:
        parser.load_previous = agentlab_data["load-existing"]
    else:
        parser.load_previous = False

    if "research-topic" in agentlab_data:
        parser.research_topic = agentlab_data["research-topic"]
    if "api-key" in agentlab_data:
        parser.api_key = agentlab_data["api-key"]
    if "deepseek-api-key" in agentlab_data:
        parser.deepseek_api_key = agentlab_data["deepseek-api-key"]

    parser.compile_latex = agentlab_data.get("compile-latex", True)
    parser.llm_backend = agentlab_data.get("llm-backend", "o3-mini")
    parser.lit_review_backend = agentlab_data.get("lit-review-backend", "gpt-4o-mini")
    parser.language = agentlab_data.get("language", "English")
    parser.num_papers_lit_review = agentlab_data.get("num-papers-lit-review", 5)
    parser.mlesolver_max_steps = agentlab_data.get("mlesolver-max-steps", 3)
    parser.papersolver_max_steps = agentlab_data.get("papersolver-max-steps", 5)
    parser.task_notes = agentlab_data.get("task-notes", [])
    parser.num_papers_to_write = agentlab_data.get("num-papers-to-write", 100)
    parser.parallel_labs = agentlab_data.get("parallel-labs", False)
    parser.num_parallel_labs = agentlab_data.get("num-parallel-labs", 8)
    parser.except_if_fail = agentlab_data.get("except-if-fail", False)
    parser.agentRxiv = agentlab_data.get("agentRxiv", False)
    parser.construct_agentRxiv = agentlab_data.get("construct-agentRxiv", False)
    parser.agentrxiv_papers = agentlab_data.get("agentrxiv-papers", 5)
    parser.lab_index = agentlab_data.get("lab-index", 0)

    return parser


if __name__ == "__main__":
    user_args = parse_arguments()
    yaml_to_use = user_args.yaml_location
    args = parse_yaml(yaml_to_use)

    llm_backend = args.llm_backend
    human_mode = args.copilot_mode.lower() == "true" if isinstance(args.copilot_mode, str) else args.copilot_mode
    compile_pdf = args.compile_latex.lower() == "true" if isinstance(args.compile_latex, str) else args.compile_latex
    load_previous = args.load_previous.lower() == "true" if isinstance(args.load_previous, str) else args.load_previous
    parallel_labs = args.parallel_labs.lower() == "true" if isinstance(args.parallel_labs, str) else args.parallel_labs
    except_if_fail = args.except_if_fail.lower() == "true" if isinstance(args.except_if_fail, str) else args.except_if_fail
    agentRxiv = args.agentRxiv.lower() == "true" if isinstance(args.agentRxiv, str) else args.agentRxiv
    construct_agentRxiv = args.construct_agentRxiv.lower() == "true" if isinstance(args.construct_agentRxiv, str) else args.construct_agentRxiv
    # FIX: check the type of args.lab_index, not args.construct_agentRxiv
    lab_index = int(args.lab_index) if isinstance(args.lab_index, str) else args.lab_index

    try:
        num_papers_to_write = int(args.num_papers_to_write.lower()) if isinstance(args.num_papers_to_write, str) else args.num_papers_to_write
    except Exception:
        raise Exception("args.num_papers_lit_review must be a valid integer!")
    try:
        num_papers_lit_review = int(args.num_papers_lit_review.lower()) if isinstance(args.num_papers_lit_review, str) else args.num_papers_lit_review
    except Exception:
        raise Exception("args.num_papers_lit_review must be a valid integer!")
    try:
        papersolver_max_steps = int(args.papersolver_max_steps.lower()) if isinstance(args.papersolver_max_steps, str) else args.papersolver_max_steps
    except Exception:
        raise Exception("args.papersolver_max_steps must be a valid integer!")
    try:
        mlesolver_max_steps = int(args.mlesolver_max_steps.lower()) if isinstance(args.mlesolver_max_steps, str) else args.mlesolver_max_steps
    except Exception:
        raise Exception("args.mlesolver_max_steps must be a valid integer!")

    if parallel_labs:
        num_parallel_labs = int(args.num_parallel_labs)
        print("=" * 20, f"RUNNING {num_parallel_labs} LABS IN PARALLEL", "=" * 20)
    else:
        num_parallel_labs = 0

    api_key = (os.getenv("OPENAI_API_KEY") or getattr(args, "api_key", None)) if (hasattr(args, "api_key") or os.getenv("OPENAI_API_KEY")) else None
    deepseek_api_key = (os.getenv("DEEPSEEK_API_KEY") or getattr(args, "deepseek_api_key", None)) if (hasattr(args, "deepseek_api_key") or os.getenv("DEEPSEEK_API_KEY")) else None
    if api_key is not None and os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if deepseek_api_key is not None and os.getenv("DEEPSEEK_API_KEY") is None:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    if not api_key and not deepseek_api_key:
        raise ValueError("API key must be provided via --api-key / -deepseek-api-key or the OPENAI_API_KEY / DEEPSEEK_API_KEY environment variable.")

    research_topic = input("Please name an experiment idea for AgentLaboratory to perform: ") if (human_mode or args.research_topic is None) else args.research_topic

    task_notes_LLM = []
    task_notes = args.task_notes
    for _task in task_notes:
        for _note in task_notes[_task]:
            task_notes_LLM.append({"phases": [_task.replace("-", " ")], "note": _note})

    if args.language != "English":
        task_notes_LLM.append({
            "phases": ["literature review", "plan formulation", "data preparation", "running experiments", "results interpretation", "report writing", "report refinement"],
            "note": f"You should always write in the following language to converse and to write the report {args.language}"
        })

    human_in_loop = {
        "literature review":      human_mode,
        "plan formulation":       human_mode,
        "data preparation":       human_mode,
        "running experiments":    human_mode,
        "results interpretation": human_mode,
        "report writing":         human_mode,
        "report refinement":      human_mode,
    }

    agent_models = {
        "literature review":      llm_backend,
        "plan formulation":       llm_backend,
        "data preparation":       llm_backend,
        "running experiments":    llm_backend,
        "report writing":         llm_backend,
        "results interpretation": llm_backend,
        "paper refinement":       llm_backend,
    }

    if parallel_labs:
        remove_figures()
        GLOBAL_AGENTRXIV = AgentRxiv()
        remove_directory(f"{RESEARCH_DIR_PATH}")
        os.mkdir(os.path.join(".", f"{RESEARCH_DIR_PATH}"))
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if not compile_pdf:
            raise Exception("PDF compilation must be used with agentRxiv!")

        def run_lab(parallel_lab_index):
            time_str = str()
            time_now = time.time()
            for _paper_index in range(num_papers_to_write):
                lab_dir = os.path.join(RESEARCH_DIR_PATH, f"research_dir_lab{parallel_lab_index}_paper{_paper_index}")
                os.mkdir(lab_dir)
                os.mkdir(os.path.join(lab_dir, "src"))
                os.mkdir(os.path.join(lab_dir, "tex"))
                lab_instance = LaboratoryWorkflow(
                    parallelized=True,
                    research_topic=research_topic,
                    notes=task_notes_LLM,
                    agent_model_backbone=agent_models,
                    human_in_loop_flag=human_in_loop,
                    openai_api_key=api_key,
                    compile_pdf=compile_pdf,
                    num_papers_lit_review=num_papers_lit_review,
                    papersolver_max_steps=papersolver_max_steps,
                    mlesolver_max_steps=mlesolver_max_steps,
                    paper_index=_paper_index,
                    lab_index=parallel_lab_index,
                    except_if_fail=except_if_fail,
                    lab_dir=lab_dir,
                    agentRxiv=True,
                    agentrxiv_papers=args.agentrxiv_papers
                )
                lab_instance.perform_research()
                time_str += str(time.time() - time_now) + " | "
                with open(f"agent_times_{parallel_lab_index}.txt", "w") as f:
                    f.write(time_str)
                time_now = time.time()

        with ThreadPoolExecutor(max_workers=num_parallel_labs) as executor:
            futures = [executor.submit(run_lab, lab_idx) for lab_idx in range(num_parallel_labs)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in lab: {e}")

        raise NotImplementedError("Todo: implement parallel labs")
    else:
        remove_figures()
        if agentRxiv:
            GLOBAL_AGENTRXIV = AgentRxiv(lab_index)
        if not agentRxiv:
            remove_directory(f"{RESEARCH_DIR_PATH}")
            os.mkdir(os.path.join(".", f"{RESEARCH_DIR_PATH}"))
        if not os.path.exists("state_saves"):
            os.mkdir(os.path.join(".", "state_saves"))
        time_str = str()
        time_now = time.time()
        for _paper_index in range(num_papers_to_write):
            lab_direct = f"{RESEARCH_DIR_PATH}/research_dir_{_paper_index}_lab_{lab_index}"
            os.mkdir(os.path.join(".", lab_direct))
            os.mkdir(os.path.join(f"./{lab_direct}", "src"))
            os.mkdir(os.path.join(f"./{lab_direct}", "tex"))
            lab = LaboratoryWorkflow(
                research_topic=research_topic,
                notes=task_notes_LLM,
                agent_model_backbone=agent_models,
                human_in_loop_flag=human_in_loop,
                openai_api_key=api_key,
                compile_pdf=compile_pdf,
                num_papers_lit_review=num_papers_lit_review,
                papersolver_max_steps=papersolver_max_steps,
                mlesolver_max_steps=mlesolver_max_steps,
                paper_index=_paper_index,
                except_if_fail=except_if_fail,
                agentRxiv=False,
                lab_index=lab_index,
                lab_dir=f"./{lab_direct}"
            )
            lab.perform_research()
            time_str += str(time.time() - time_now) + " | "
            with open(f"agent_times_{lab_index}.txt", "w") as f:
                f.write(time_str)
            time_now = time.time()
