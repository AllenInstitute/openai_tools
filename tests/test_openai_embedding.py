

from papers_extractor.openai_parsers import OpenaiLongParser
import os
import openai
import logging
import sys

# Replace with your own OpenAI API key or set the OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]


def test_api_embedding_call():
    openai_long_parser = OpenaiLongParser("Test prompt")
    response = openai_long_parser.call_embeddingGPT(
        "Hello World.")
    assert len(response) == 1536



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_api_embedding_call()

