import openai
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import os
from dotenv import load_dotenv
import argparse
import tiktoken

# Load the environment variables from the .env file
load_dotenv()

# Replace with your own OpenAI API key or set the OPENAI_API_KEY environment variable
openai.api_key =  os.getenv('OPENAI_API_KEY')

def count_tokens(texts):
    """Counts the number of tokens in a list of texts.
    Args:
        texts (list): A list of texts.
    Returns:
        int: The number of tokens.
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    num_tokens = 0

    for message in texts:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += len(encoding.encode(message))
        
    return num_tokens

def custom_word_tokenize(text):
    """Tokenizes a string.
    Args:
        text (str): The text to tokenize.
    Returns:
        list: A list of tokens.
    """
    lines = text.splitlines(True)
    tokens = []
    # We tokenize each line separately to keep the line breaks as tokens.
    for line in lines:
        line_tokens = word_tokenize(line.strip())
        tokens.extend(line_tokens)
        tokens.append('\n')
    return tokens[:-1]  # Remove the last newline token

def break_up_file(tokens, chunk_size, overlap_size):
    """Breaks up a file into chunks of tokens.
    Args:
        tokens (list): A list of tokens.
        chunk_size (int): The number of tokens in each chunk.
        overlap_size (int): The number of tokens to overlap between chunks.
    Returns:
        list: A list of lists of tokens.
    """
    def find_sentence_boundary(tokens, start_idx):
        # This is to only cut at the end of the last full sentence.
        current_length = start_idx
        # We go backward until we find a sentence boundary.
        for idx, token in enumerate(tokens[start_idx::-1]):
            if token in {'.', '!', '?'}:
                current_length = start_idx - idx
                break    
        return current_length

    if len(tokens) <= chunk_size:
        yield tokens
    else:
        end_idx = find_sentence_boundary(tokens, chunk_size)
        chunk = tokens[:end_idx]
        yield chunk
        yield from break_up_file(tokens[end_idx - overlap_size:], chunk_size, overlap_size)

def break_up_file_to_chunks(text, chunk_size=2000, overlap_size=0):
    """Breaks up a file into chunks of tokens.
    Args:
        text (str): The text to break up.
        chunk_size (int): The number of tokens in each chunk.
        overlap_size (int): The number of tokens to overlap between chunks.
    Returns:
        list: A list of lists of tokens.
    """
    tokens = custom_word_tokenize(text)
    return list(break_up_file(tokens, chunk_size, overlap_size))

def convert_to_detokenized_text(tokenized_text):
    """Converts a list of tokens to a detokenized string.
    Args:
        tokenized_text (list): A list of tokens.
        Returns:
        str: A detokenized string.
        """
    
    prompt_text = "".join(tokenized_text)

    return prompt_text

def call_chatGPT(prompt, temperature=0.1):
    """Calls the OpenAI API to generate text.
    Args:
        prompt (str): The prompt to use for the API call.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature to use for the API call.
    Returns:
        str: The generated text.
    """

    print("Calling the OpenAI API")
    # Nb of tokens in the prompt
    NbTokensInPrompt = count_tokens([prompt])
    print(f"Number of tokens in the prompt: {NbTokensInPrompt}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000 - NbTokensInPrompt,
        n=1,
        stop=None,
        temperature=temperature,
    )
    
    # Nb to tokens in the response
    NbTokensInResponse = count_tokens([response.choices[0].message.content])
    print(f"Number of tokens in the response: {NbTokensInResponse}")

    # Total number of tokens
    TotalNbTokens = NbTokensInPrompt + NbTokensInResponse
    print(f"Total number of tokens: {TotalNbTokens}")

    # We error out if the response was stopped before the end.
    if response.choices[0].finish_reason == "length":
        raise Exception("We stopped because we didn't reach the end of the LLM text.")

    return response.choices[0].message.content

def process_long_text_through_openai_into_chunks(prompt, text, chunk_size):
    """Processes a long text through the OpenAI API.
    Args:
        prompt (str): The prompt to use for the API call.
        text (str): The text to process.
        chunk_size (int): The number of tokens in each chunk.
    Returns:
        str: The processed text.
    """
    
    list_chunk = break_up_file_to_chunks(text, chunk_size=chunk_size)
    all_processed = []
    print(f"Number of chunks to compress: {len(list_chunk)}")
    for i, chunk in enumerate(list_chunk):
        local_text = convert_to_detokenized_text(chunk)
        print(f"Cleaning up chunk {i}/ {len(list_chunk)}")
        submit_prompt = prompt + local_text

        result = call_chatGPT(submit_prompt)

        all_processed.append(result)

    concatenated_summaries = "".join(all_processed)

    return concatenated_summaries

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
    print(f"Number of chunks to summarize: {len(list_chunk)}")
    for i, chunk in enumerate(list_chunk):
        local_text = convert_to_detokenized_text(chunk)

        print(f"Summarizing chunk {i}/ {len(list_chunk)}")
        # We take 100 characters from the previous paragraph
        prompt = "Write a long, very detailed summary for a technical expert\
                of the following paragraph, from a paper, refering to the text as -This publication-:\n" \
                + local_text

        result = call_chatGPT(prompt)

        all_summaries.append(result)

    concatenated_summaries = "\n".join(all_summaries)
    nb_chunks = len(list_chunk)
    return concatenated_summaries, nb_chunks

if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_pdf', 
                        help='Path to a pdf file', 
                        type=str, 
                        default=os.path.join(script_path, '../example/2020.12.15.422967v4.full.pdf'))
    parser.add_argument('--save_summary', 
                        help='Save the summary in a txt file along the pdf file', 
                        type=bool, 
                        default=True)
    parser.add_argument('--save_raw_text', 
                        help='Save the raw text in a txt file along the pdf file', 
                        type=bool, 
                        default=False)
    parser.add_argument('--save_compressed_text',
                        help='Save the compressed text in a txt file along the pdf file',
                        type=bool,
                        default=False)
    parser.add_argument('--cut_bibliography',
                        help='Try not to summarize the bibliography at the end of the pdf file',
                        type=bool,
                        default=True)
   
    parser.add_argument('--chunk_length',
                        help='This is to increase the final length of the summary. The document is summarized in chunks. More \
                        chunks means a longer summary. Inconsitencty across the sections could occur with larger number. Typically \
                        1 is a good value for an abstract and 2, 3 for more details.',
                        type=int,
                        default=1)
    
    # Here we can add a flag to select the section to summarize
    # these are the sections we can select from
    # abstract, introduction, methods, results, discussion
    parser.add_argument('--select_section',
                        help='Select the section to summarize. The options are: all(default), introduction, methods, results, discussion',
                        type=str,
                        choices= ['all', 'introduction', 'methods', 'results', 'discussion'], 
                        default='all')
    args = parser.parse_args()

    pdf_path = args.path_pdf
    save_summary = args.save_summary
    save_raw_text = args.save_raw_text
    save_compressed_text = args.save_compressed_text
    chunk_length = args.chunk_length
    select_section = args.select_section

    # Extract text from the PDF
    laparams = LAParams()
    text = extract_text(pdf_path, laparams=laparams)

    # We save the raw text in a txt file
    if save_raw_text:
        raw_text_path = pdf_path.replace(".pdf", "_raw.txt")
        with open(raw_text_path, "w") as f:
            f.write(text)   

    # We remove the bibliography
    # This is a very low tech way to do it, we will improve it later
    # Bibliography is usually at the end of the document
    # And it creates large number of tokens
    if args.cut_bibliography:
        if "References" in text:
            text = text.split("References")[0]
        if "Bibliography" in text:
            text = text.split("Bibliography")[0]

    # We first clean up the text
    print("Cleaning up and compressing the text")
    prompt = "Remove formatting text, the abstract, author list, figure captions, references & bibliography, page number, headers and footers from the following text from a scientific publication. Don't change any other words:"
    text_compressed = process_long_text_through_openai_into_chunks(prompt, text, chunk_size=1500)

    # We save the raw text in a txt file
    if save_compressed_text:
        raw_text_path = pdf_path.replace(".pdf", "_compressed.txt")
        with open(raw_text_path, "w") as f:
            f.write(text_compressed)   

    # We summarize the text into chunks
    nb_chunks = 10

    current_text = text_compressed
    while (nb_chunks > chunk_length):        
        current_text, nb_chunks = summarize_text_into_chunks(current_text)
        print(f"Current number of chunks:{nb_chunks}")
    
    # We can afford to clean up if the text is not too long
    if ((nb_chunks > 1) & (count_tokens([current_text]) < 2000)):
        print("Cleaning up the summary")
        # We count the number of tokens
        # If it small enough, we send the text for a last clean up. 
        prompt = "Can you clean up this publication summary to make it flow logically. Keep this summary very technical and detailed:\n" \
                + current_text

        current_text = call_chatGPT(prompt)

    # We save the summary in a txt file
    if save_summary:
        summary_path = pdf_path.replace(".pdf", ".txt")
        with open(summary_path, "w") as f:
            f.write(current_text)

    # We print the final summary
    print(current_text)
