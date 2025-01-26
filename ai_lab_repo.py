from agents import *
from copy import copy
from common_imports import *
from mlesolver import MLESolver
from torch.backends.mkl import verbose

import argparse
import pickle

DEFAULT_LLM_BACKBONE = "o1-mini"


class LaboratoryWorkflow:
    def __init__(self, research_topic, openai_api_key, max_steps=100, num_papers_lit_review=5, agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}", notes=list(), human_in_loop_flag=None, compile_pdf=True, mlesolver_max_steps=3, papersolver_max_steps=5):
        """
        Initialize laboratory workflow
        @param research_topic: (str) description of research idea to explore
        @param max_steps: (int) max number of steps for each phase, i.e. compute tolerance budget
        @param num_papers_lit_review: (int) number of papers to include in the lit review
        @param agent_model_backbone: (str or dict) model backbone to use for agents
        @param notes: (list) notes for agent to follow during tasks
        """

        self.notes = notes
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.openai_api_key = openai_api_key
        self.research_topic = research_topic
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review

        self.print_cost = True
        self.review_override = True # should review be overridden?
        self.review_ovrd_steps = 0 # review steps so far
        self.arxiv_paper_exp_time = 3
        self.reference_papers = list()

        ##########################################
        ####### COMPUTE BUDGET PARAMETERS ########
        ##########################################
        self.num_ref_papers = 1
        self.review_total_steps = 0 # num steps to take if overridden
        self.arxiv_num_summaries = 5
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps

        self.phases = [
            ("literature review", ["literature review"]),
            ("plan formulation", ["plan formulation"]),
            ("experimentation", ["data preparation", "running experiments"]),
            ("results interpretation", ["results interpretation", "report writing", "report refinement"]),
        ]
        self.phase_status = dict()
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        self.phase_models = dict()
        if type(agent_model_backbone) == str:
            for phase, subtasks in self.phases:
                for subtask in subtasks:
                    self.phase_models[subtask] = agent_model_backbone
        elif type(agent_model_backbone) == dict:
            # todo: check if valid
            self.phase_models = agent_model_backbone


        self.human_in_loop_flag = human_in_loop_flag

        self.statistics_per_phase = {
            "literature review":      {"time": 0.0, "steps": 0.0,},
            "plan formulation":       {"time": 0.0, "steps": 0.0,},
            "data preparation":       {"time": 0.0, "steps": 0.0,},
            "running experiments":    {"time": 0.0, "steps": 0.0,},
            "results interpretation": {"time": 0.0, "steps": 0.0,},
            "report writing":         {"time": 0.0, "steps": 0.0,},
            "report refinement":      {"time": 0.0, "steps": 0.0,},
        }

        self.save = True
        self.verbose = True
        self.reviewers = ReviewersAgent(model=self.model_backbone, notes=self.notes, openai_api_key=self.openai_api_key)
        self.phd = PhDStudentAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.postdoc = PostdocAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.professor = ProfessorAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.ml_engineer = MLEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.sw_engineer = SWEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)

        # remove previous files
        remove_figures()
        remove_directory("research_dir")
        # make src and research directory
        if not os.path.exists("state_saves"):
            os.mkdir(os.path.join(".", "state_saves"))
        os.mkdir(os.path.join(".", "research_dir"))
        os.mkdir(os.path.join("./research_dir", "src"))
        os.mkdir(os.path.join("./research_dir", "tex"))

    def set_model(self, model):
        self.set_agent_attr("model", model)
        self.reviewers.model = model

    def save_state(self, phase):
        """
        Save state for phase
        @param phase: (str) phase string
        @return: None
        """
        phase = phase.replace(" ", "_")
        with open(f"state_saves/{phase}.pkl", "wb") as f:
            pickle.dump(self, f)

    def set_agent_attr(self, attr, obj):
        """
        Set attribute for all agents
        @param attr: (str) agent attribute
        @param obj: (object) object attribute
        @return: None
        """
        setattr(self.phd, attr, obj)
        setattr(self.postdoc, attr, obj)
        setattr(self.professor, attr, obj)
        setattr(self.ml_engineer, attr, obj)
        setattr(self.sw_engineer, attr, obj)

    def reset_agents(self):
        """
        Reset all agent states
        @return: None
        """
        self.phd.reset()
        self.postdoc.reset()
        self.professor.reset()
        self.ml_engineer.reset()
        self.sw_engineer.reset()

    def perform_research(self):
        """
        Loop through all research phases
        @return: None
        """
        for phase, subtasks in self.phases:
            phase_start_time = time.time()  # Start timing the phase
            if self.verbose: print(f"{'*'*50}\nBeginning phase: {phase}\n{'*'*50}")
            for subtask in subtasks:
                if self.verbose: print(f"{'&'*30}\nBeginning subtask: {subtask}\n{'&'*30}")
                if type(self.phase_models) == dict:
                    if subtask in self.phase_models:
                        self.set_model(self.phase_models[subtask])
                    else: self.set_model(f"{DEFAULT_LLM_BACKBONE}")
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "literature review":
                    repeat = True
                    while repeat: repeat = self.literature_review()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "plan formulation":
                    repeat = True
                    while repeat: repeat = self.plan_formulation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "data preparation":
                    repeat = True
                    while repeat: repeat = self.data_preparation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "running experiments":
                    repeat = True
                    while repeat: repeat = self.running_experiments()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "results interpretation":
                    repeat = True
                    while repeat: repeat = self.results_interpretation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report writing":
                    repeat = True
                    while repeat: repeat = self.report_writing()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report refinement":
                    return_to_exp_phase = self.report_refinement()

                    if not return_to_exp_phase:
                        if self.save: self.save_state(subtask)
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
                if self.save: self.save_state(subtask)
                # Calculate and print the duration of the phase
                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time
                print(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                self.statistics_per_phase[subtask]["time"] = phase_duration

    def report_refinement(self):
        """
        Perform report refinement phase
        @return: (bool) whether to repeat the phase
        """
        reviews = self.reviewers.inference(self.phd.plan, self.phd.report)
        print("Reviews:", reviews)
        if self.human_in_loop_flag["report refinement"]:
            print(f"Provided are reviews from a set of three reviewers: {reviews}")
            input("Would you like to be completed with the project or should the agents go back and improve their experimental results?\n (y) for go back (n) for complete project: ")
        else:
            review_prompt = f"Provided are reviews from a set of three reviewers: {reviews}. Would you like to be completed with the project or do you want to go back to the planning phase and improve your experiments?\n Type y and nothing else to go back, type n and nothing else for complete project."
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
                if verbose: print("*"*40, "\n", "REVIEW COMPLETE", "\n", "*"*40)
                return False
            elif response == "y":
                self.set_agent_attr("reviewer_response", f"Provided are reviews from a set of three reviewers: {reviews}.")
                return True
            else: raise Exception("Model did not respond")

    def report_writing(self):
        """
        Perform report writing phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        report_notes = [_note["note"] for _note in self.ml_engineer.notes if "report writing" in _note["phases"]]
        report_notes = f"Notes for the task objective: {report_notes}\n" if len(report_notes) > 0 else ""
        # instantiate mle-solver
        from papersolver import PaperSolver
        self.reference_papers = []
        solver = PaperSolver(notes=report_notes, max_steps=self.papersolver_max_steps, plan=lab.phd.plan, exp_code=lab.phd.results_code, exp_results=lab.phd.exp_results, insights=lab.phd.interpretation, lit_review=lab.phd.lit_review, ref_papers=self.reference_papers, topic=research_topic, openai_api_key=self.openai_api_key, llm_str=self.model_backbone["report writing"], compile_pdf=compile_pdf)
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.papersolver_max_steps):
            solver.solve()
        # get best report results
        report = "\n".join(solver.best_report[0][0])
        score = solver.best_report[0][1]
        if self.verbose: print(f"Report writing completed, reward function score: {score}")
        if self.human_in_loop_flag["report writing"]:
            retry = self.human_in_loop("report writing", report)
            if retry: return retry
        self.set_agent_attr("report", report)
        readme = self.professor.generate_readme()
        save_to_file("./research_dir", "readme.md", readme)
        save_to_file("./research_dir", "report.txt", report)
        self.reset_agents()
        return False

    def results_interpretation(self):
        """
        Perform results interpretation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            resp = self.postdoc.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)
            if "```INTERPRETATION" in resp:
                interpretation = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag["results interpretation"]:
                    retry = self.human_in_loop("results interpretation", interpretation)
                    if retry: return retry
                self.set_agent_attr("interpretation", interpretation)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["results interpretation"]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        raise Exception("Max tries during phase: Results Interpretation")

    def running_experiments(self):
        """
        Perform running experiments phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        experiment_notes = [_note["note"] for _note in self.ml_engineer.notes if "running experiments" in _note["phases"]]
        experiment_notes = f"Notes for the task objective: {experiment_notes}\n" if len(experiment_notes) > 0 else ""
        # instantiate mle-solver
        solver = MLESolver(dataset_code=self.ml_engineer.dataset_code, notes=experiment_notes, insights=self.ml_engineer.lit_review_sum, max_steps=self.mlesolver_max_steps, plan=self.ml_engineer.plan, openai_api_key=self.openai_api_key, llm_str=self.model_backbone["running experiments"])
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.mlesolver_max_steps-1):
            solver.solve()
        # get best code results
        code = "\n".join(solver.best_codes[0][0])
        # regenerate figures from top code
        execute_code(code)
        score = solver.best_codes[0][1]
        exp_results = solver.best_codes[0][2]
        if self.verbose: print(f"Running experiments completed, reward function score: {score}")
        if self.human_in_loop_flag["running experiments"]:
            retry = self.human_in_loop("data preparation", code)
            if retry: return retry
        save_to_file("./research_dir/src", "run_experiments.py", code)
        self.set_agent_attr("results_code", code)
        self.set_agent_attr("exp_results", exp_results)
        # reset agent state
        self.reset_agents()
        return False

    def data_preparation(self):
        """
        Perform data preparation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        ml_feedback = str()
        ml_dialogue = str()
        swe_feedback = str()
        ml_command = str()
        hf_engine = HFDataSearch()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            if ml_feedback != "":
                ml_feedback_in = "Feedback provided to the ML agent: " + ml_feedback
            else: ml_feedback_in = ""
            resp = self.sw_engineer.inference(self.research_topic, "data preparation", feedback=f"{ml_dialogue}\nFeedback from previous command: {swe_feedback}\n{ml_command}{ml_feedback_in}", step=_i)
            swe_feedback = str()
            swe_dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                swe_dialogue = f"\nThe following is dialogue produced by the SW Engineer: {dialogue}\n"
                if self.verbose: print("#"*40, f"\nThe following is dialogue produced by the SW Engineer: {dialogue}", "\n", "#"*40)
            if "```SUBMIT_CODE" in resp:
                final_code = extract_prompt(resp, "SUBMIT_CODE")
                code_resp = execute_code(final_code)
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")
                swe_feedback += f"\nCode Response: {code_resp}\n"
                if "[CODE EXECUTION ERROR]" in code_resp:
                    swe_feedback += "\nERROR: Final code had an error and could not be submitted! You must address and fix this error.\n"
                else:
                    if self.human_in_loop_flag["data preparation"]:
                        retry = self.human_in_loop("data preparation", final_code)
                        if retry: return retry
                    save_to_file("./research_dir/src", "load_data.py", final_code)
                    self.set_agent_attr("dataset_code", final_code)
                    # reset agent state
                    self.reset_agents()
                    self.statistics_per_phase["data preparation"]["steps"] = _i
                    return False

            if ml_feedback != "":
                ml_feedback_in = "Feedback from previous command: " + ml_feedback
            else:
                ml_feedback_in = ""
            resp = self.ml_engineer.inference(
                self.research_topic, "data preparation",
                feedback=f"{swe_dialogue}\n{ml_feedback_in}", step=_i)
            #if self.verbose: print("ML Engineer: ", resp, "\n~~~~~~~~~~~")
            ml_feedback = str()
            ml_dialogue = str()
            ml_command = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                ml_dialogue = f"\nThe following is dialogue produced by the ML Engineer: {dialogue}\n"
                if self.verbose: print("#" * 40, f"\nThe following is dialogue produced by the ML Engineer: {dialogue}", "#" * 40, "\n")
            if "```python" in resp:
                code = extract_prompt(resp, "python")
                code = self.ml_engineer.dataset_code + "\n" + code
                code_resp = execute_code(code)
                ml_command = f"Code produced by the ML agent:\n{code}"
                ml_feedback += f"\nCode Response: {code_resp}\n"
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")
            if "```SEARCH_HF" in resp:
                hf_query = extract_prompt(resp, "SEARCH_HF")
                hf_res = "\n".join(hf_engine.results_str(hf_engine.retrieve_ds(hf_query)))
                ml_command = f"HF search command produced by the ML agent:\n{hf_query}"
                ml_feedback += f"Huggingface results: {hf_res}\n"
        raise Exception("Max tries during phase: Data Preparation")

    def plan_formulation(self):
        """
        Perform plan formulation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            # inference postdoc to
            resp = self.postdoc.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                if self.human_in_loop_flag["plan formulation"]:
                    retry = self.human_in_loop("plan formulation", plan)
                    if retry: return retry
                self.set_agent_attr("plan", plan)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["plan formulation"]["steps"] = _i
                return False

            resp = self.phd.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")

            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        raise Exception("Max tries during phase: Plan Formulation")

    def literature_review(self):
        """
        Perform literature review phase
        @return: (bool) whether to repeat the phase
        """
        arx_eng = ArxivSearch()
        max_tries = self.max_steps * 5  # lit review often requires extra steps
        
        # get initial response from PhD agent
        resp = self.phd.inference(self.research_topic, "literature review", step=0, temp=0.8)
        if self.verbose:
            print(resp, "\n~~~~~~~~~~~")

        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            feedback = str()

            # grab summary of papers from arxiv
            if "```SUMMARY" in resp:
                query = extract_prompt(resp, "SUMMARY")
                papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                feedback = f"You requested arXiv papers related to the query {query}, here was the response\n{papers}"

            # grab full text from arxiv ID
            elif "```FULL_TEXT" in resp:
                query = extract_prompt(resp, "FULL_TEXT")
                try:
                    # expiration timer so that paper does not remain in context too long
                    full_text_content = arx_eng.retrieve_full_paper_text(query)
                except Exception as e:
                    # Catch any unexpected errors from arxiv.Client()
                    err_msg = f"[ERROR] Could not retrieve paper. Possibly invalid arXiv ID. Error: {e}"
                    full_text_content = err_msg

                # In case retrieve_full_paper_text returns an error string
                # or if we want to unify it, e.g. "[ERROR] ...something"
                if full_text_content.startswith("[ERROR]"):
                    # We won't crash; just pass that as feedback
                    arxiv_paper = f"```EXPIRATION {self.arxiv_paper_exp_time}\n{full_text_content}```"
                else:
                    # normal successful retrieval
                    arxiv_paper = f"```EXPIRATION {self.arxiv_paper_exp_time}\n{full_text_content}```"

                feedback = arxiv_paper

            # if add paper, extract and add to lit review, provide feedback
            elif "```ADD_PAPER" in resp:
                query = extract_prompt(resp, "ADD_PAPER")
                feedback, text = self.phd.add_review(query, arx_eng)
                # If we want to store reference text for later usage
                if len(self.reference_papers) < self.num_ref_papers:
                    self.reference_papers.append(text)

            # completion condition
            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                # generate formal review
                lit_review_sum = self.phd.format_review()

                # if human in loop -> check if human is happy with the produced review
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    # if not happy, repeat the process with human feedback
                    if retry:
                        self.phd.lit_review = []
                        return retry

                # otherwise, return lit review and move on to next stage
                if self.verbose:
                    print(self.phd.lit_review_sum)
                # set agent
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False

            # Move on to the next iteration with new feedback
            resp = self.phd.inference(
                self.research_topic, 
                "literature review", 
                feedback=feedback, 
                step=_i + 1, 
                temp=0.8
            )
            if self.verbose:
                print(resp, "\n~~~~~~~~~~~")

        # If we exceed max_tries:
        raise Exception("Max tries during phase: Literature Review")


    def human_in_loop(self, phase, phase_prod):
        """
        Get human feedback for phase output
        @param phase: (str) current phase
        @param phase_prod: (str) current phase result
        @return: (bool) whether to repeat the loop
        """
        print("\n\n\n\n\n")
        print(f"Presented is the result of the phase [{phase}]: {phase_prod}")
        y_or_no = None
        # repeat until a valid answer is provided
        while y_or_no not in ["y", "n"]:
            y_or_no = input("\n\n\nAre you happy with the presented content? Respond Y or N: ").strip().lower()
            # if person is happy with feedback, move on to next stage
            if y_or_no == "y": pass
            # if not ask for feedback and repeat
            elif y_or_no == "n":
                # ask the human for feedback
                notes_for_agent = input("Please provide notes for the agent so that they can try again and improve performance: ")
                # reset agent state
                self.reset_agents()
                # add suggestions to the notes
                self.notes.append({
                    "phases": [phase],
                    "note": notes_for_agent})
                return True
            else: print("Invalid response, type Y or N")
        return False



def parse_arguments():
    parser = argparse.ArgumentParser(description="AgentLaboratory Research Workflow")

    parser.add_argument(
        '--copilot-mode',
        type=str,
        default="False",
        help='Enable human interaction mode.'
    )

    parser.add_argument(
        '--deepseek-api-key',
        type=str,
        help='Provide the DeepSeek API key.'
    )

    parser.add_argument(
        '--load-existing',
        type=str,
        default="False",
        help='Do not load existing state; start a new workflow.'
    )

    parser.add_argument(
        '--load-existing-path',
        type=str,
        help='Path to load existing state; start a new workflow, e.g. state_saves/results_interpretation.pkl'
    )

    parser.add_argument(
        '--research-topic',
        type=str,
        help='Specify the research topic.'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='Provide the OpenAI API key.'
    )

    parser.add_argument(
        '--compile-latex',
        type=str,
        default="True",
        help='Compile latex into pdf during paper writing phase. Disable if you can not install pdflatex.'
    )

    parser.add_argument(
        '--llm-backend',
        type=str,
        default="o1-mini",
        help='Backend LLM to use for agents in Agent Laboratory.'
    )

    parser.add_argument(
        '--language',
        type=str,
        default="English",
        help='Language to operate Agent Laboratory in.'
    )

    parser.add_argument(
        '--num-papers-lit-review',
        type=str,
        default="5",
        help='Total number of papers to summarize in literature review stage'
    )

    parser.add_argument(
        '--mlesolver-max-steps',
        type=str,
        default="3",
        help='Total number of mle-solver steps'
    )

    parser.add_argument(
        '--papersolver-max-steps',
        type=str,
        default="5",
        help='Total number of paper-solver steps'
    )

    parser.add_argument('--file-path', type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    llm_backend = args.llm_backend
    human_mode = args.copilot_mode.lower() == "true"
    compile_pdf = args.compile_latex.lower() == "true"
    load_existing = args.load_existing.lower() == "true"
    file_path = args.file_path

    try:
        num_papers_lit_review = int(args.num_papers_lit_review.lower())
    except Exception:
        raise Exception("args.num_papers_lit_review must be a valid integer!")
    try:
        papersolver_max_steps = int(args.papersolver_max_steps.lower())
    except Exception:
        raise Exception("args.papersolver_max_steps must be a valid integer!")
    try:
        mlesolver_max_steps = int(args.mlesolver_max_steps.lower())
    except Exception:
        raise Exception("args.papersolver_max_steps must be a valid integer!")


    api_key = os.getenv('OPENAI_API_KEY') or args.api_key
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY') or args.deepseek_api_key
    if args.api_key is not None and os.getenv('OPENAI_API_KEY') is None:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.deepseek_api_key is not None and os.getenv('DEEPSEEK_API_KEY') is None:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    if not api_key and not deepseek_api_key:
        raise ValueError("API key must be provided via --api-key / -deepseek-api-key or the OPENAI_API_KEY / DEEPSEEK_API_KEY environment variable.")

    ##########################################################
    # Research question that the agents are going to explore #
    ##########################################################
    if human_mode or args.research_topic is None:
        research_topic = input("Please name an experiment idea for AgentLaboratory to perform: ")
    else:
        research_topic = args.research_topic

    if file_path and "{FILE}" in research_topic:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        # Replace the placeholder with the entire file text
        research_topic = research_topic.replace("{FILE}", file_content)


    task_notes_LLM = [
        {"phases": ["plan formulation"],
         "note": f"You should come up with a plan for TWO experiments."},

        {"phases": ["plan formulation", "data preparation", "running experiments"],
         "note": "Please use gpt-4o-mini for your experiments."},

        {"phases": ["running experiments"],
         "note": f'Use the following code to inference gpt-4o-mini: \nfrom openai import OpenAI\nos.environ["OPENAI_API_KEY"] = "{api_key}"\nclient = OpenAI()\ncompletion = client.chat.completions.create(\nmodel="gpt-4o-mini-2024-07-18", messages=messages)\nanswer = completion.choices[0].message.content\n'},

        {"phases": ["running experiments"],
         "note": f"You have access to only gpt-4o-mini using the OpenAI API, please use the following key {api_key} but do not use too many inferences. Do not use openai.ChatCompletion.create or any openai==0.28 commands. Instead use the provided inference code."},

        {"phases": ["running experiments"],
         "note": "I would recommend using a small dataset (approximately only 100 data points) to run experiments in order to save time. Do not use much more than this unless you have to or are running the final tests."},

        {"phases": ["data preparation", "running experiments"],
         "note": "You are running on a MacBook laptop. You can use 'mps' with PyTorch"},

        {"phases": ["data preparation", "running experiments"],
         "note": "Generate figures with very colorful and artistic design."},
    ]

    task_notes_LLM.append(
        {"phases": ["literature review", "plan formulation", "data preparation", "running experiments", "results interpretation", "report writing", "report refinement"],
        "note": f"You should always write in the following language to converse and to write the report {args.language}"},
    )

    ####################################################
    ###  Stages where human input will be requested  ###
    ####################################################
    human_in_loop = {
        "literature review":      human_mode,
        "plan formulation":       human_mode,
        "data preparation":       human_mode,
        "running experiments":    human_mode,
        "results interpretation": human_mode,
        "report writing":         human_mode,
        "report refinement":      human_mode,
    }

    ###################################################
    ###  LLM Backend used for the different phases  ###
    ###################################################
    agent_models = {
        "literature review":      llm_backend,
        "plan formulation":       llm_backend,
        "data preparation":       llm_backend,
        "running experiments":    llm_backend,
        "report writing":         llm_backend,
        "results interpretation": llm_backend,
        "paper refinement":       llm_backend,
    }

    if load_existing:
        load_path = args.load_existing_path
        if load_path is None:
            raise ValueError("Please provide path to load existing state.")
        with open(load_path, "rb") as f:
            lab = pickle.load(f)
    else:
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
        )

    lab.perform_research()






