

from papers_extractor.openai_parsers import \
    count_tokens, custom_word_tokenize, custom_word_detokenize
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


def test_counter():
    assert count_tokens(["Test prompt"]) == 6


def test_custom_word_tokenize():
    test_str = "Test prompt. Hello world!"
    question = custom_word_tokenize(test_str)
    response = ['Test', 'prompt', '.', 'Hello', 'world', '!']
    assert question == response


def test_custom_word_tokenize_detokenize():
    # We create multiple sentences that explore the different cases
    # With many different characters.
    list_sentences_to_test = [
        "Test prompt. Hello world!",
        "Hello, I am a man. Who are?",
        "I am making weird punctuations.?",
        "I can speak, but I can't write. Where is the problem?",
        "What happens with long words like antidisestablishmentarianism?",
        "What happens with common long words like constitution?",
        "Did you know, she asked with a smile, that I've visited over " +
        "15 countries in the past year?",
        "From Paris to Tokyo, Rio to Sydney, it's been an amazing journey; " +
        "and the memories will last a lifetime.",
        "Wait, hold on – wasn't that the guy from the movie we saw last " +
        "week... or am I mistaken?",
        "In the box, I found a variety of items: pens, pencils, erasers, " +
        "paper clips, and a small stapler.",
        "He shouted, Fire! but nobody reacted; they all thought it was just " +
        "another drill.",
        "I'm not sure if I should take the job in London, she mused, or the " +
        "one in New York.",
        "My favorite fruits are apples, bananas, and oranges; however, I " +
        "also enjoy the occasional strawberry or grape.",
        "When the clock struck midnight, we all clinked our glasses together" +
        ", cheering: Happy New Year!",
        "As the sun set, the sky turned a brilliant array of colors – pink, " +
        "orange, and purple – and the stars began to appear one by one.",
        "He whispered, I love you, as they embraced beneath the moonlit sky," +
        " surrounded by the soft glow of fireflies."]

    for test_str in list_sentences_to_test:
        back_str = custom_word_detokenize(custom_word_tokenize(test_str))
        assert back_str == test_str


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_counter()
    test_custom_word_tokenize()
    test_custom_word_tokenize_detokenize()
    # We know there is a problem with sentences that contains quotes like "
