

from papers_extractor.openai_parsers import OpenaiLongParser
import os
import openai
import logging
import sys

# Import the dotenv module to load the environment variables
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Replace with your own OpenAI API key or set the OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]


def test_api_embedding_call():
    openai_long_parser = OpenaiLongParser("Test prompt")
    response = openai_long_parser.call_embeddingGPT(
        "Hello World.")
    assert len(response) == 1536


def test_api_chunk_embedding_call():
    test_str = 'Hello World! \
        Hello World!'
    openai_long_parser = OpenaiLongParser(test_str, chunk_size=3)
    response = openai_long_parser.process_chunks_through_embedding()
    assert len(response) == 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_api_embedding_call()
    test_api_chunk_embedding_call()
