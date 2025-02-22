# General-purpose imports
import os
import sys
import json
import time
import re
import math
import logging
import random
import shutil
import pathlib
import argparse
import itertools
import datetime
import collections
import subprocess

# Data manipulation and analysis
import pandas as pd
import numpy as np
import csv
import json
import yaml
import h5py
import sqlite3
import pickle

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Hugging Face & Transformers
import transformers

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchaudio
import tensorflow as tf
#import keras

# NLP Libraries
import tiktoken
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
import sacremoses
# Diffusers for image generation and stable diffusion
import diffusers
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Performance acceleration libraries
import accelerate
from accelerate import Accelerator

# Hugging Face Hub utilities
import huggingface_hub
from huggingface_hub import HfApi, notebook_login

# Scikit-learn for machine learning
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# Statistical analysis
import scipy
from scipy import stats, signal, spatial
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean, cosine
from scipy.linalg import svd, eig
from statsmodels.api import OLS, Logit
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, pacf, acf

# Image processing and handling
from PIL import Image
import imageio
from skimage import io, color, filters, transform, exposure

# File handling and I/O
import gzip
import zipfile
import tarfile
import glob

# Parallel processing
import multiprocessing
from multiprocessing import Pool

# Miscellaneous utilities
import hashlib
import uuid
import base64
import warnings
from tqdm import tqdm
from functools import partial, lru_cache

# Other advanced libraries
import pydantic
import requests
import aiohttp
