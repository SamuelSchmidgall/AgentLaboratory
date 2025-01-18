from utils import *

import time
import arxiv
import os, re
import io, sys
import numpy as np
import concurrent.futures
from pypdf import PdfReader
from datasets import load_dataset
from psutil._common import bytes2human
from datasets import load_dataset_builder
from semanticscholar import SemanticScholar
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

import traceback
import concurrent.futures
import psutil
import subprocess

class HFDataSearch:
    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        """
        Class for finding relevant huggingface datasets
        :param like_thr: threshold of 'likes'
        :param dwn_thr: threshold of 'downloads'
        """
        self.dwn_thr = dwn_thr
        self.like_thr = like_thr
        self.ds = load_dataset("nkasmanoff/huggingface-datasets")["train"]

        # Initialize lists to collect filtered data
        filtered_indices = []
        filtered_descriptions = []
        filtered_likes = []
        filtered_downloads = []

        # Iterate over the dataset and filter based on criteria
        for idx, item in enumerate(self.ds):
            # Get likes and downloads, handling None values
            likes = int(item['likes']) if item['likes'] is not None else 0
            downloads = int(item['downloads']) if item['downloads'] is not None else 0

            # Check if likes and downloads meet the thresholds
            if likes >= self.like_thr and downloads >= self.dwn_thr:
                # Check if the description is a non-empty string
                description = item['description']
                if isinstance(description, str) and description.strip():
                    # Collect the data
                    filtered_indices.append(idx)
                    filtered_descriptions.append(description)
                    filtered_likes.append(likes)
                    filtered_downloads.append(downloads)

        # Check if any datasets meet all criteria
        if not filtered_indices:
            print("No datasets meet the specified criteria.")
            self.ds = []
            self.descriptions = []
            self.likes_norm = []
            self.downloads_norm = []
            self.description_vectors = None
            return  # Exit the constructor

        # Filter the datasets using the collected indices
        self.ds = self.ds.select(filtered_indices)

        # Update descriptions, likes, and downloads
        self.descriptions = filtered_descriptions
        self.likes = np.array(filtered_likes)
        self.downloads = np.array(filtered_downloads)

        # Normalize likes and downloads
        self.likes_norm = self._normalize(self.likes)
        self.downloads_norm = self._normalize(self.downloads)

        # Vectorize the descriptions
        self.vectorizer = TfidfVectorizer()
        self.description_vectors = self.vectorizer.fit_transform(self.descriptions)

    def _normalize(self, arr):
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr, dtype=float)
        return (arr - min_val) / (max_val - min_val)

    def retrieve_ds(self, query, N=10, sim_w=1.0, like_w=0.0, dwn_w=0.0):
        """
        Retrieves the top N datasets matching the query, weighted by likes and downloads.
        :param query: The search query string.
        :param N: The number of results to return.
        :param sim_w: Weight for cosine similarity.
        :param like_w: Weight for likes.
        :param dwn_w: Weight for downloads.
        :return: List of top N dataset items.
        """
        if not self.ds or self.description_vectors is None:
            print("No datasets available to search.")
            return []

        query_vector = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vector, self.description_vectors).flatten()
        # Normalize cosine similarities
        cosine_similarities_norm = self._normalize(cosine_similarities)

        # Compute final scores
        final_scores = (
            sim_w * cosine_similarities_norm +
            like_w * self.likes_norm +
            dwn_w * self.downloads_norm
        )

        # Get top N indices
        top_indices = final_scores.argsort()[-N:][::-1]
        # Convert indices to Python ints
        top_indices = [int(i) for i in top_indices]
        top_datasets = [self.ds[i] for i in top_indices]

        # Check if dataset has a test & train set; gather size info
        has_test_set = []
        has_train_set = []
        ds_size_info = []
        for i in top_indices:
            try:
                dbuilder = load_dataset_builder(self.ds[i]["id"], trust_remote_code=True).info
            except Exception as e:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue

            if dbuilder.splits is None:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue

            has_test, has_train = "test" in dbuilder.splits, "train" in dbuilder.splits
            has_test_set.append(has_test)
            has_train_set.append(has_train)

            test_dwn_size, test_elem_size = None, None
            train_dwn_size, train_elem_size = None, None
            if has_test:
                test_dwn_size = bytes2human(dbuilder.splits["test"].num_bytes)
                test_elem_size = dbuilder.splits["test"].num_examples
            if has_train:
                train_dwn_size = bytes2human(dbuilder.splits["train"].num_bytes)
                train_elem_size = dbuilder.splits["train"].num_examples

            ds_size_info.append((test_dwn_size, test_elem_size, train_dwn_size, train_elem_size))

        # Attach metadata to the top_datasets
        for _i in range(len(top_datasets)):
            top_datasets[_i]["has_test_set"] = has_test_set[_i]
            top_datasets[_i]["has_train_set"] = has_train_set[_i]
            top_datasets[_i]["test_download_size"] = ds_size_info[_i][0]
            top_datasets[_i]["test_element_size"] = ds_size_info[_i][1]
            top_datasets[_i]["train_download_size"] = ds_size_info[_i][2]
            top_datasets[_i]["train_element_size"] = ds_size_info[_i][3]

        return top_datasets

    def results_str(self, results):
        """
        Provide results as list of results in human-readable format.
        :param results: (list(dict)) list of results from search
        :return: (list(str)) list of results in human-readable format
        """
        result_strs = []
        for result in results:
            res_str = f"Dataset ID: {result['id']}\n"
            res_str += f"Description: {result['description']}\n"
            res_str += f"Likes: {result['likes']}\n"
            res_str += f"Downloads: {result['downloads']}\n"
            res_str += f"Has Testing Set: {result['has_test_set']}\n"
            res_str += f"Has Training Set: {result['has_train_set']}\n"
            res_str += f"Test Download Size: {result['test_download_size']}\n"
            res_str += f"Test Dataset Size: {result['test_element_size']}\n"
            res_str += f"Train Download Size: {result['train_download_size']}\n"
            res_str += f"Train Dataset Size: {result['train_element_size']}\n"
            result_strs.append(res_str)
        return result_strs

class SemanticScholarSearch:
    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)

    def find_papers_by_str(self, query, N=10):
        """
        Finds top-N papers from semantic scholar
        :param query: str
        :param N: number of results
        :return: list of string summaries
        """
        paper_sums = []
        results = self.sch_engine.search_paper(
            query, 
            limit=N, 
            min_citation_count=3, 
            open_access_pdf=True
        )
        for _i in range(len(results)):
            paper_sum = f"Title: {results[_i].title}\n"
            paper_sum += f"Abstract: {results[_i].abstract}\n"
            paper_sum += f"Citations: {results[_i].citationCount}\n"
            paper_sum += (
                f"Release Date: year {results[_i].publicationDate.year}, "
                f"month {results[_i].publicationDate.month}, "
                f"day {results[_i].publicationDate.day}\n"
            )
            paper_sum += f"Venue: {results[_i].venue}\n"
            paper_sum += f"Paper ID: {results[_i].externalIds['DOI']}\n"
            paper_sums.append(paper_sum)
        return paper_sums

    def retrieve_full_paper_text(self, query):
        """
        NOTE: Not implemented in this example
        """
        pass

class ArxivSearch:
    def __init__(self):
        # Construct the default API client.
        self.sch_engine = arxiv.Client()
        
    def _process_query(self, query: str) -> str:
        """
        Process query string to fit within MAX_QUERY_LENGTH 
        while preserving as much info as possible
        """
        MAX_QUERY_LENGTH = 300
        if len(query) <= MAX_QUERY_LENGTH:
            return query

        # Split into words
        words = query.split()
        processed_query = []
        current_length = 0

        # Add words while staying under the limit
        for word in words:
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break
        return ' '.join(processed_query)
    
    def find_papers_by_str(self, query, N=20):
        """
        Finds top-N relevant arXiv papers
        """
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=N,
                    sort_by=arxiv.SortCriterion.Relevance
                )

                paper_sums = []
                for r in self.sch_engine.results(search):
                    paperid = r.pdf_url.split("/")[-1]
                    pubdate = str(r.published).split(" ")[0]
                    paper_sum = f"Title: {r.title}\n"
                    paper_sum += f"Summary: {r.summary}\n"
                    paper_sum += f"Publication Date: {pubdate}\n"
                    paper_sum += f"Categories: {' '.join(r.categories)}\n"
                    paper_sum += f"arXiv paper ID: {paperid}\n"
                    paper_sums.append(paper_sum)
                time.sleep(2.0)
                return "\n".join(paper_sums)

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    # Exponential-ish back-off
                    time.sleep(2 * retry_count)
                    continue
        
        # If unsuccessful
        return None

    def retrieve_full_paper_text(self, query):
        """
        Download and extract full text from arXiv PDF
        """
        pdf_text = ""
        # Attempt to get single result with the provided paper ID
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))
        
        # Download the PDF to a local file
        paper.download_pdf(filename="downloaded-paper.pdf")

        # Create a pdf reader object
        reader = PdfReader("downloaded-paper.pdf")

        # Iterate over pages
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
            except Exception:
                os.remove("downloaded-paper.pdf")
                time.sleep(2.0)
                return "EXTRACTION FAILED"

            pdf_text += f"--- Page {page_number} ---"
            pdf_text += text
            pdf_text += "\n"

        os.remove("downloaded-paper.pdf")
        time.sleep(2.0)
        return pdf_text


def execute_code(
    code_str,
    max_total_time=7200,   # Hard limit on total runtime (seconds)
    max_idle_time=60,      # If no CPU usage / no prints for this many secs, kill
    max_stdout_len=2000
):
    """
    Execute code in a separate subprocess with:
      1) absolute time limit (max_total_time)
      2) idle time limit (max_idle_time) based on CPU usage
      3) capture stdout (limited by max_stdout_len)
      4) gracefully handle zombie processes or processes that vanish
    """
    import matplotlib
    matplotlib.use('Agg')  # Use a non-interactive backend
    import matplotlib.pyplot as plt

    # Basic checks
    if "load_dataset('pubmed" in code_str:
        return "[CODE EXECUTION ERROR] pubmed Download took way too long. Program terminated"
    if "exit(" in code_str:
        return "[CODE EXECUTION ERROR] The exit() command is not allowed; please remove it."

    temp_filename = "temp_script.py"
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(code_str)

    start_time = time.time()
    output_capture = io.StringIO()

    process = subprocess.Popen(
        [sys.executable, temp_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    proc_psutil = psutil.Process(process.pid)
    last_output_time = time.time()
    last_cpu_check_time = time.time()
    kill_reason = None

    # For versions of psutil which do have STATUS_ZOMBIE:
    # We'll prefer that, else fallback to the literal "zombie".
    ZOMBIE_STRING = getattr(psutil, "STATUS_ZOMBIE", "zombie").lower()

    while True:
        # 1) If the process ended, break the loop
        if process.poll() is not None:
            break

        # 2) Read any line from stdout
        try:
            line = process.stdout.readline()
        except Exception:
            line = None

        if line:
            output_capture.write(line)
            last_output_time = time.time()

        # 3) CPU usage check every 2 seconds
        if (time.time() - last_cpu_check_time) > 2.0:
            last_cpu_check_time = time.time()

            # (a) If process is not running => presumably done or zombie
            if not proc_psutil.is_running():
                kill_reason = "Process ended or unknown state"
                break

            # (b) Attempt to get the string status
            try:
                # e.g. "running", "sleeping", "zombie", etc.
                status_str = proc_psutil.status().lower()
            except psutil.Error:
                # If we can't retrieve status, assume it's gone
                kill_reason = "Could not retrieve process status"
                process.kill()
                break

            # If status is "zombie", kill it
            if status_str == ZOMBIE_STRING:
                kill_reason = "Process is zombie"
                process.kill()
                break

            # (c) Try CPU usage
            try:
                cpu_usage = proc_psutil.cpu_percent(interval=0.1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                kill_reason = "Process is gone or cannot be accessed"
                process.kill()
                break

            # If usage > 1%, consider it active
            if cpu_usage > 1.0:
                last_output_time = time.time()

        # 4) Idle check
        if (time.time() - last_output_time) > max_idle_time:
            kill_reason = f"Idle for {max_idle_time} seconds."
            process.kill()
            break

        # 5) Total time check
        if (time.time() - start_time) > max_total_time:
            kill_reason = f"Exceeded total runtime of {max_total_time} seconds."
            process.kill()
            break

        time.sleep(0.05)  # Short pause

    # 6) Drain leftover stdout
    if process.poll() is not None:
        leftover = process.stdout.read()
        if leftover:
            output_capture.write(leftover)

    process.stdout.close()
    process.wait()

    # 7) Remove temp file if desired
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    # 8) Attach kill_reason to logs if any
    output = output_capture.getvalue()
    if kill_reason:
        output += f"\n[CODE EXECUTION STOPPED]: {kill_reason}\n"

    return output[:max_stdout_len]


