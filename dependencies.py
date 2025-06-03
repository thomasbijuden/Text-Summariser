import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

dir(nltk)