import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

def tfidf_summarizer(text, num_sentences=3):
    """
    Summarizes text using TF-IDF.

    Args:
        text (str): The input text to summarize.
        num_sentences (int): The desired number of sentences in the summary.

    Returns:
        str: The generated summary.
        None: If the text is too short or processing fails.
    """
    if not text.strip():
        print("Warning: Input text is empty.")
        return ""

    original_sentences, processed_sentences_tokens = preprocess_text(text)

    if not original_sentences or not processed_sentences_tokens:
        print("Warning: No processable sentences found in the text.")
        return text # Return original text if no valid sentences after processing

    if len(original_sentences) <= num_sentences:
        print(f"Warning: Input text has {len(original_sentences)} sentences, which is less than or equal to the requested summary length of {num_sentences}. Returning original text.")
        return " ".join(original_sentences)

    # Join tokens back into strings for TfidfVectorizer
    processed_sentences_str = [" ".join(tokens) for tokens in processed_sentences_tokens]

    # 1. Calculate TF-IDF
    # Each sentence is treated as a 'document' for IDF calculation
    try:
        vectorizer = TfidfVectorizer(smooth_idf=True) # smooth_idf adds 1 to document frequencies, preventing zero divisions
        tfidf_matrix = vectorizer.fit_transform(processed_sentences_str)
    except ValueError as e:
        if "empty vocabulary" in str(e):
            print("Warning: TF-IDF vocabulary is empty. This might happen if all words are stop words or too short.")
            # Fallback: return top sentences by order
            return " ".join(original_sentences[:num_sentences])
        else:
            raise e


    # 2. Score Sentences
    # The score of a sentence is the sum of TF-IDF scores of its words
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    if len(sentence_scores) == 0:
        print("Warning: Could not calculate sentence scores. Returning original text.")
        return " ".join(original_sentences)


    # 3. Select Top N Sentences
    # Get indices of sentences with highest scores
    # Add a small value proportional to position to favor earlier sentences in case of tie
    # This helps maintain some coherence if scores are very similar.
    # The factor is small so it doesn't dominate TF-IDF scores.
    positional_factor = 0.0001 
    ranked_sentence_indices = [
        i for i, score in sorted(
            enumerate(sentence_scores),
            key=lambda x: x[1] - (x[0] * positional_factor), # Higher score is better, earlier sentence is slightly preferred
            reverse=True
        )
    ]
    
    # Ensure we don't request more sentences than available after scoring
    num_to_select = min(num_sentences, len(ranked_sentence_indices))
    top_sentence_indices = sorted(ranked_sentence_indices[:num_to_select])


    # 4. Form Summary
    summary = " ".join([original_sentences[i] for i in top_sentence_indices])
    return summary

if __name__ == '__main__':
    example_text = (
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
        "concerned with the interactions between computers and human language, in particular how to program computers "
        "to process and analyze large amounts of natural language data. Challenges in NLP frequently involve speech "
        "recognition, natural language understanding, and natural language generation. Text summarization is one of the "
        "key applications of NLP. It aims to create a concise and fluent summary of a longer text document. "
        "Automatic text summarization methods are greatly needed to address the ever-increasing amount of text data "
        "available online. Simple techniques like TF-IDF can be quite effective for extractive summarization."
    )

    summary1 = tfidf_summarizer(example_text, num_sentences=2)
    print("\n--- TF-IDF Summary (2 sentences, no stemming) ---")
    print(summary1)

    short_text = "This is a sentence. This is another sentence that is also very important."
    summary_short = tfidf_summarizer(short_text, num_sentences=1)
    print("\n--- TF-IDF Summary for short text (1 sentence) ---")
    print(summary_short)
    
    all_stopwords_text = "Is it the an of by."
    summary_stopwords = tfidf_summarizer(all_stopwords_text, num_sentences=1)
    print("\n--- TF-IDF Summary for all stopwords text ---")
    print(summary_stopwords) # Expected to be empty or problematic

    empty_text = ""
    summary_empty = tfidf_summarizer(empty_text, num_sentences=1)
    print("\n--- TF-IDF Summary for empty text ---")
    print(summary_empty)