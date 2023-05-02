from papers_extractor.long_paper import LongPaper
import logging
import sys
import openai
import os
import numpy as np

# Import the dotenv module to load the environment variables
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Replace with your own OpenAI API key or set the OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]

def test_tsne_plot_longpaper():
    # This is mostly a smoke test to see if the plot is generated
    list_sentences_to_test = [
        "The vast expanse of the cosmos never ceases to astonish us.\
            The universe is a testament to the beauty and wonder of creation. \
            Scientists continue to uncover new celestial phenomena. \
            We are reminded of the enormity of existence. \
            Our humble place within it.",
    ]
    list_sentences_to_test = "\n".join(list_sentences_to_test) 
    long_paper = LongPaper(list_sentences_to_test, chunk_size=10)
    figure_handle = long_paper.plot_tsne_embedding(perplexity=3)
    assert figure_handle is not None

def test_average_embedding():
    list_sentences_to_test = \
        "The vast expanse of the cosmos never ceases to astonish us."

    long_paper = LongPaper(list_sentences_to_test, chunk_size=3)
    average_embedding = long_paper.get_average_embedding()
    assert len(average_embedding)==1536

def test_calculate_embedding():
    list_sentences_to_test = \
        "The vast expanse of the cosmos never ceases to astonish us."

    long_paper = LongPaper(list_sentences_to_test, chunk_size=3)
    long_paper.calculate_embedding()

    assert np.array(long_paper.embedding).shape==(4, 1536)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_tsne_plot_longpaper()
    test_average_embedding()
    test_calculate_embedding()