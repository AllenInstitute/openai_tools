

from papers_extractor.openai_parsers import OpenaiLongParser
import os
import openai
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Replace with your own OpenAI API key or set the OPENAI_API_KEY
# environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# We first test a call to the API


def test_api_call():
    openai_long_parser = OpenaiLongParser("Test prompt")
    response = openai_long_parser.call_chatGPT(
        "Say Hello World, I am a test.", temperature=0)
    assert response == 'Hello World, I am a test.'
