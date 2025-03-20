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


class HFDataSearch:
    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        """
        Class for finding relevant huggingface datasets
        :param like_thr:
        :param dwn_thr:
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
        # check if dataset has a test & train set
        has_test_set = list()
        has_train_set = list()
        ds_size_info = list()
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
            # Print number of examples for
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
        result_strs = list()
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
        paper_sums = list()
        results = self.sch_engine.search_paper(query, limit=N, min_citation_count=3, open_access_pdf=True)
        for _i in range(len(results)):
            paper_sum = f'Title: {results[_i].title}\n'
            paper_sum += f'Abstract: {results[_i].abstract}\n'
            paper_sum += f'Citations: {results[_i].citationCount}\n'
            paper_sum += f'Release Date: year {results[_i].publicationDate.year}, month {results[_i].publicationDate.month}, day {results[_i].publicationDate.day}\n'
            paper_sum += f'Venue: {results[_i].venue}\n'
            paper_sum += f'Paper ID: {results[_i].externalIds["DOI"]}\n'
            paper_sums.append(paper_sum)
        return paper_sums

    def retrieve_full_paper_text(self, query):
        pass


class ArxivSearch:
    def __init__(self):
        # Construct the default API client.
        self.sch_engine = arxiv.Client()
        
    def _process_query(self, query: str) -> str:
        """Process query string to fit within MAX_QUERY_LENGTH while preserving as much information as possible"""
        MAX_QUERY_LENGTH = 300
        
        if len(query) <= MAX_QUERY_LENGTH:
            return query
        
        # Split into words
        words = query.split()
        processed_query = []
        current_length = 0
        
        # Add words while staying under the limit
        # Account for spaces between words
        for word in words:
            # +1 for the space that will be added between words
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break
            
        return ' '.join(processed_query)
    
    def find_papers_by_str(self, query, N=20):
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=N,
                    sort_by=arxiv.SortCriterion.Relevance)

                paper_sums = list()
                # `results` is a generator; you can iterate over its elements one by one...
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
                    # 递增延时
                    time.sleep(2 * retry_count)
                    continue
                
        return None

    def retrieve_full_paper_text(self, query):
        pdf_text = str()
        pdf_filename = "downloaded-paper.pdf"  # Temporary PDF filename

        try:
            # Fetch the paper
            try:
                paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))
            except StopIteration:
                print(f"No results found for query: {query}")
                return "NO RESULTS FOUND"
            except arxiv.HTTPError as e:
                print(f"Failed to fetch paper {query}: {e}")
                return f"HTTP ERROR: {e}"
            except Exception as e:
                print(f"An unexpected error occurred while fetching paper: {e}")
                return f"UNEXPECTED ERROR: {e}"

            # Download the PDF file
            try:
                paper.download_pdf(filename=pdf_filename)
            except Exception as e:
                print(f"Failed to download PDF for paper {query}: {e}")
                return f"DOWNLOAD ERROR: {e}"

            # Read the PDF file and extract text
            try:
                reader = PdfReader(pdf_filename)
                for page_number, page in enumerate(reader.pages, start=1):
                    try:
                        text = page.extract_text()
                        pdf_text += f"--- Page {page_number} ---\n"
                        pdf_text += text
                        pdf_text += "\n"
                    except Exception as e:
                        print(f"Failed to extract text from page {page_number}: {e}")
                        pdf_text += f"--- Page {page_number} ---\n"
                        pdf_text += "EXTRACTION FAILED\n"
            except Exception as e:
                print(f"Failed to read PDF file: {e}")
                return f"PDF READ ERROR: {e}"

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return f"UNEXPECTED ERROR: {e}"
        finally:
            # Clean up temporary files
            if os.path.exists(pdf_filename):
                try:
                    os.remove(pdf_filename)
                except Exception as e:
                    print(f"Failed to delete temporary file {pdf_filename}: {e}")
            time.sleep(2.0)  # Avoid frequent requests

        return pdf_text

"""
import multiprocessing
import sys
import io
import traceback

def execute_code(code_str, timeout=180):
    if "load_dataset('pubmed" in code_str:
        return "pubmed Download took way too long. Program terminated"

    def run_code(queue):
        # Redirect stdout to capture print outputs
        output_capture = io.StringIO()
        sys.stdout = output_capture

        try:
            exec_globals = {}
            exec(code_str, exec_globals)
        except Exception as e:
            output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
            traceback.print_exc(file=output_capture)
        finally:
            # Put the output in the queue
            queue.put(output_capture.getvalue())
            # Restore stdout
            sys.stdout = sys.__stdout__

    # Create a multiprocessing Queue to capture the output
    queue = multiprocessing.Queue()
    # Create a new Process
    process = multiprocessing.Process(target=run_code, args=(queue,))
    process.start()
    # Wait for the process to finish or timeout
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds. You must reduce the time complexity of your code."
    else:
        # Retrieve the output from the queue
        output = queue.get()
        return output

"""

import io
import sys
import traceback
import concurrent.futures



import multiprocessing
import io
import sys
import traceback
import multiprocessing
import io
import sys
import traceback


def execute_code(code_str, timeout=60, MAX_LEN=1000):
    #print(code_str)

    # prevent plotting errors
    import matplotlib
    matplotlib.use('Agg')  # Use the non-interactive Agg backend
    import matplotlib.pyplot as plt

    # Preventing execution of certain resource-intensive datasets
    if "load_dataset('pubmed" in code_str:
        return "[CODE EXECUTION ERROR] pubmed Download took way too long. Program terminated"
    if "exit(" in code_str:
        return "[CODE EXECUTION ERROR] The exit() command is not allowed you must remove this."
    #print(code_str)
    # Capturing the output
    output_capture = io.StringIO()
    sys.stdout = output_capture

    # Create a new global context for exec
    exec_globals = globals()

    def run_code():
        try:
            # Executing the code in the global namespace
            exec(code_str, exec_globals)
        except Exception as e:
            output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
            traceback.print_exc(file=output_capture)

    try:
        # Running code in a separate thread with a timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_code)
            future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds. You must reduce the time complexity of your code."
    except Exception as e:
        return f"[CODE EXECUTION ERROR]: {str(e)}"
    finally:
        # Restoring standard output
        sys.stdout = sys.__stdout__

    # Returning the captured output
    return output_capture.getvalue()[:MAX_LEN]



