from papers_extractor.long_paper import LongPaper
import logging
import sys
import os
import openai
# Import the dotenv module to load the environment variables
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Replace with your own OpenAI API key or set the OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]

# We first test the LongPaper class for a short text


def test_content_database():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext)
    long_paper_obj.save_database()
    key_iterator = long_paper_obj.database.iterkeys()
    key_list = list(key_iterator)
    assert long_paper_obj.database_id in key_list

def test_long_paper_key():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext)
    local_key = long_paper_obj.database_id
    assert local_key == "a54d88e06612d820bc3be72877c74f257b561b196f34a3e0e" + \
    "1af181e8a78e70c146682b7ead12846"

def test_long_paper_saving_database():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext)
    long_paper_obj.save_database()
    assert long_paper_obj.database_id in long_paper_obj.database

def test_custom_long_paper_key():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext, database_id="custom_key")
    local_key = long_paper_obj.database_id
    assert local_key == "custom_key"
    long_paper_obj.save_database()
    assert long_paper_obj.database_id in long_paper_obj.database

def test_long_paper_embedding_caching():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext)
    embedding = long_paper_obj.calculate_embedding()
    long_paper_obj.save_database()
    embedding = long_paper_obj.calculate_embedding()
    databased_data = long_paper_obj.database[long_paper_obj.database_id]

    assert "embedding" in databased_data
    assert databased_data['embedding'] == embedding

def test_long_paper_summarizing_caching():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext)
    summary = long_paper_obj.summarize_longtext_into_chunks()
    long_paper_obj.save_database()
    summary = long_paper_obj.summarize_longtext_into_chunks()
    databased_data = long_paper_obj.database[long_paper_obj.database_id]

    assert "summary" in databased_data
    assert databased_data['summary'] == summary
    
def test_reset_database():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext)
    long_paper_obj.save_database()
    check_in = long_paper_obj.database_id in long_paper_obj.database
    assert check_in == True
    long_paper_obj.reset_database()
    check_in = long_paper_obj.database_id in long_paper_obj.database
    assert check_in == False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_long_paper_key()
    test_long_paper_saving_database()
    test_custom_long_paper_key()
    test_long_paper_embedding_caching()
    test_long_paper_summarizing_caching()
    test_content_database()
    test_reset_database()