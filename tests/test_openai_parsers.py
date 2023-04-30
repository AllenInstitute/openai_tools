

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

# We first test a call to the API


def test_api_call():
    openai_long_parser = OpenaiLongParser("Test prompt")
    response = openai_long_parser.multi_call_chatGPT(
        ["Say Hello World, I am a test."], temperature=0)
    assert response == ['Hello World, I am a test.']


def test_break_up_veryshortsentence_to_chunks():
    test_str = 'Hello World'

    openai_long_parser = OpenaiLongParser(test_str, chunk_size=3)
    response = ['Hello World']
    assert openai_long_parser.chunks == response


def test_break_up_veryshortendedsentence_to_chunks():
    test_str = 'Hello World.'

    openai_long_parser = OpenaiLongParser(test_str, chunk_size=3)
    response = ['Hello World.']
    assert openai_long_parser.chunks == response


def test_break_up_shortsentences_to_chunks():
    test_str = 'Hello World. \
        Goodbye World.'

    openai_long_parser = OpenaiLongParser(test_str, chunk_size=3)
    response = ['Hello World.',
                'Goodbye World.']
    assert openai_long_parser.chunks == response


def test_break_up_longsentences_to_chunks():
    test_str = 'Test prompt for the first sentence. \
        Hello world in the second sentence.'

    openai_long_parser = OpenaiLongParser(test_str, chunk_size=10)
    response = ['Test prompt for the first sentence.',
                'Hello world in the second sentence.']
    print(openai_long_parser.chunks)

    assert openai_long_parser.chunks == response


def test_break_up_threesentences_to_chunks():
    test_str = 'Test prompt for the first sentence. \
        Hello world in the second sentence. \
        Goodbye world in the third sentence.'

    openai_long_parser = OpenaiLongParser(test_str, chunk_size=10)
    response = ['Test prompt for the first sentence.',
                'Hello world in the second sentence.',
                'Goodbye world in the third sentence.']
    print(openai_long_parser.chunks)

    assert openai_long_parser.chunks == response


def test_break_up_groupsentences_to_chunks():
    test_str = 'Test prompt for the first sentence. \
        Hello world in the second sentence. \
        Goodbye world in the third sentence.'

    openai_long_parser = OpenaiLongParser(test_str, chunk_size=15)
    response = [
        'Test prompt for the first sentence. ' +
        'Hello world in the second sentence.',
        'Goodbye world in the third sentence.']
    print(openai_long_parser.chunks)

    assert openai_long_parser.chunks == response


def test_break_up_unfinishedsentences_to_chunks():
    test_str = 'Test prompt for the first sentence. \
        Hello world in the second sentence. \
        Goodbye world in the third sentence'

    openai_long_parser = OpenaiLongParser(test_str, chunk_size=10)
    response = ['Test prompt for the first sentence.',
                'Hello world in the second sentence.',
                'Goodbye world in the third sentence']
    print(openai_long_parser.chunks)

    assert openai_long_parser.chunks == response


def test_break_up_unfinishedgroupsentences_to_chunks():
    test_str = 'Test prompt for the first sentence. \
        Hello world in the second sentence. \
        Goodbye world in the third sentence'

    openai_long_parser = OpenaiLongParser(test_str, chunk_size=15)
    response = [
        'Test prompt for the first sentence. ' +
        'Hello world in the second sentence.',
        'Goodbye world in the third sentence']
    print(openai_long_parser.chunks)
    assert openai_long_parser.chunks == response


def test_process_chunks_through_prompt():
    test_str = 'Hello World! \
        Hello World!'
    openai_long_parser = OpenaiLongParser(test_str, chunk_size=3)
    prompt = "Say"
    reply = openai_long_parser.process_chunks_through_prompt(prompt,
                                                             temperature=0)
    response = ['Hello World!',
                'Hello World!']
    assert reply == response


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_api_call()
    test_break_up_veryshortsentence_to_chunks()
    test_break_up_veryshortendedsentence_to_chunks()
    test_break_up_shortsentences_to_chunks()
    test_break_up_longsentences_to_chunks()
    test_break_up_threesentences_to_chunks()
    test_break_up_groupsentences_to_chunks()
    test_break_up_unfinishedsentences_to_chunks()
    test_break_up_unfinishedgroupsentences_to_chunks()
    test_process_chunks_through_prompt()
