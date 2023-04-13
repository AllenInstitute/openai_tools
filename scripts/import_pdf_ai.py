import openai
import requests
from io import BytesIO
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Replace with your own OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Replace with the path to your PDF
pdf_path = r'PATH_TO_PDF'

def count_tokens(text):
    tokens = word_tokenize(text)
    return len(tokens)

def break_up_file(tokens, chunk_size, overlap_size):
    """Breaks up a file into chunks of tokens.
    Args:
        tokens (list): A list of tokens.
        chunk_size (int): The number of tokens in each chunk.
        overlap_size (int): The number of tokens to overlap between chunks.
    Returns:
        list: A list of lists of tokens.
    """

    if len(tokens) <= chunk_size:
        yield tokens
    else:
        chunk = tokens[:chunk_size]
        yield chunk
        yield from break_up_file(tokens[chunk_size-overlap_size:], chunk_size, overlap_size)

def break_up_file_to_chunks(text, chunk_size=1000, overlap_size=100):
    """Breaks up a file into chunks of tokens.
    Args:
        text (str): The text to break up.
        chunk_size (int): The number of tokens in each chunk.
        overlap_size (int): The number of tokens to overlap between chunks.
    Returns:
        list: A list of lists of tokens.
    """

    tokens = word_tokenize(text)
    return list(break_up_file(tokens, chunk_size, overlap_size))

def convert_to_detokenized_text(tokenized_text):
    """Converts a list of tokens to a detokenized string.
    Args:
        tokenized_text (list): A list of tokens.
        Returns:
        str: A detokenized string.
        """
    
    prompt_text = " ".join(tokenized_text)
    prompt_text = prompt_text.replace(" 's", "'s")
    return prompt_text

# Extract text from the PDF
laparams = LAParams()
text = extract_text(pdf_path, laparams=laparams)

# We cut the abstract
text = text[1595:]

def summarize_text_into_chunks(text):
    """Summarizes a text into chunks.
    Args:
        text (str): The text to summarize. 
    Returns:
        str: The summarized text.
        int: The number of chunks.
    """

    all_summaries = []
    list_chunk = break_up_file_to_chunks(text)
    for i, chunk in enumerate(list_chunk):
        local_text = convert_to_detokenized_text(chunk)

        prompt = "Write a summary for a technical expert: " + local_text
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Print the summary
        all_summaries.append(response.choices[0].message.content)
        # print(response.choices[0].text.strip())

    concatenated_summaries = " ".join(all_summaries)

    nb_chunks = len(list_chunk)
    return concatenated_summaries, nb_chunks

# We summarize the text into chunks
nb_chunks = 10
current_text = text
while (nb_chunks > 1):
    print(nb_chunks)
    current_text, nb_chunks = summarize_text_into_chunks(current_text)

# We print the final summary
print(current_text)