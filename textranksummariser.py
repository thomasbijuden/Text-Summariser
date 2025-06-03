import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import networkx as nx
# from collections import Counter


# # Download necessary NLTK data if not already present
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt', quiet=True)
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords', quiet=True)


def preprocess_text(text):
    """
    Preprocesses the text:
    - Sentence tokenization
    - Word tokenization for each sentence
    - Lowercasing
    - Stop word removal
    - Punctuation removal
    Returns a list of original sentences and a list of processed sentences (lists of words).
    """
    original_sentences = sent_tokenize(text)
    processed_sentences_tokens = []
    stop_words = set(stopwords.words('english'))

    for sentence in original_sentences:
        # Remove punctuation and convert to lowercase
        cleaned_sentence = re.sub(r'[^\w\s]', '', sentence).lower()
        words = word_tokenize(cleaned_sentence)
        
        # Remove stop words and optionally stem
        filtered_words = []
        for word in words:
            if word not in stop_words and word.strip(): # ensure word is not empty
                filtered_words.append(word)
        
        if filtered_words: # Only add if there are non-stop words
            processed_sentences_tokens.append(filtered_words)
        elif original_sentences: # If all words were stop words, keep original sentence for scoring later but with empty tokens
             processed_sentences_tokens.append([])


    # Ensure original_sentences and processed_sentences_tokens align if some sentences became empty
    # This basic version will just use non-empty processed sentences.
    # A more robust version might map scores back more carefully if empty processed sentences occur.
    
    valid_indices = [i for i, tokens in enumerate(processed_sentences_tokens) if tokens]
    aligned_original_sentences = [original_sentences[i] for i in valid_indices]
    aligned_processed_sentences_tokens = [processed_sentences_tokens[i] for i in valid_indices]
    


    return aligned_original_sentences, aligned_processed_sentences_tokens


def calculate_sentence_similarity(tokens1, tokens2):
    """
    Calculates Jaccard similarity between two lists of tokens.
    """
    if not tokens1 or not tokens2:
        return 0.0
    
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    return intersection / union


def textrank_summarizer_nltk(text, num_sentences=3):
    """
    Summarizes text using TextRank with NLTK for preprocessing.

    Args:
        text (str): The input text to summarize.
        num_sentences (int): The desired number of sentences in the summary.
        use_stemming (bool): Whether to use PorterStemmer.

    Returns:
        str: The generated summary.
        None: If text is too short or processing fails.
    """
    if not text.strip():
        print("Warning: Input text is empty.")
        return ""

    original_sentences, processed_tokens_lists = preprocess_text(text)

    if not original_sentences or not processed_tokens_lists:
        print("Warning: No processable sentences found in the text.")
        return text # Return original if nothing useful came out of preprocessing

    if len(original_sentences) <= num_sentences:
        print(f"Warning: Input text has {len(original_sentences)} sentences, which is less than or equal to the requested summary length of {num_sentences}. Returning original text.")
        return " ".join(original_sentences)

    # 1. Build Similarity Graph
    similarity_graph = nx.Graph()

    # Add nodes (sentences by their index in original_sentences)
    for i in range(len(processed_tokens_lists)):
        similarity_graph.add_node(i)

    # Add edges with similarity scores
    for i in range(len(processed_tokens_lists)):
        for j in range(i + 1, len(processed_tokens_lists)):
            similarity = calculate_sentence_similarity(
                processed_tokens_lists[i], 
                processed_tokens_lists[j]
            )
            if similarity > 0.01: # Add edge if similarity is above a threshold
                similarity_graph.add_edge(i, j, weight=similarity)
    
    if not similarity_graph.nodes or not similarity_graph.edges:
        print("Warning: Could not build a meaningful similarity graph. Returning top sentences by order.")
        summary_sentences_texts = [original_sentences[i] for i in range(min(num_sentences, len(original_sentences)))]
        return " ".join(summary_sentences_texts)

    # 2. Rank sentences using PageRank
    try:
        scores = nx.pagerank(similarity_graph, weight='weight')
    except nx.PowerIterationFailedConvergence:
        print("Warning: PageRank did not converge. Using simple sentence order as fallback.")
        summary_sentences_texts = [original_sentences[i] for i in range(min(num_sentences, len(original_sentences)))]
        return " ".join(summary_sentences_texts)
    except Exception as e: # Catch other potential networkx errors
        print(f"Error during PageRank calculation: {e}. Using simple sentence order as fallback.")
        summary_sentences_texts = [original_sentences[i] for i in range(min(num_sentences, len(original_sentences)))]
        return " ".join(summary_sentences_texts)


    # 3. Select Top N Sentences
    ranked_sentence_indices = sorted(
        ((scores[i], i) for i in scores), 
        reverse=True
    )
    
    # Ensure we don't request more sentences than available
    num_to_select = min(num_sentences, len(ranked_sentence_indices))
    
    top_sentence_indices = sorted(
        [idx for score, idx in ranked_sentence_indices[:num_to_select]]
    )

    # 4. Form Summary
    summary = " ".join([original_sentences[i] for i in top_sentence_indices])
    return summary

if __name__ == '__main__':
    example_text_nltk = (
        "Automated text summarization is a key technology in natural language processing. "
        "Its goal is to produce a concise and fluent summary of a longer text document. "
        "Many algorithms exist, including graph-based methods like TextRank. "
        "TextRank builds a graph where sentences are nodes and edges represent similarity. "
        "The PageRank algorithm is then applied to this graph to score the sentences. "
        "Sentences with higher scores are considered more important for the summary."
    )


    summary_nltk_nostem = textrank_summarizer_nltk(example_text_nltk, num_sentences=3)
    print("\n--- TextRank NLTK Summary (3 sentences) ---")
    print(summary_nltk_nostem)

    short_text_nltk = "This is a sentence. This is another sentence. A third one perhaps."
    summary_short_nltk = textrank_summarizer_nltk(short_text_nltk, num_sentences=3)
    print("\n--- TextRank NLTK Summary for short text (3 sentences) ---")
    print(summary_short_nltk)

    very_short_text_nltk = "One sentence only."
    summary_veryshort_nltk = textrank_summarizer_nltk(very_short_text_nltk, num_sentences=1)
    print("\n--- TextRank NLTK Summary for very short text (1 sentence) ---")
    print(summary_veryshort_nltk)

    empty_text_val = ""
    summary_empty_val = textrank_summarizer_nltk(empty_text_val, num_sentences=1)
    print("\n--- TextRank NLTK Summary for empty text ---")
    print(summary_empty_val)