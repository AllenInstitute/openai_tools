from papers_extractor.long_paper import LongPaper
from papers_extractor.database_parser import LocalDatabase
import logging
import sys
import os
import openai
import pytest

# Import the dotenv module to load the environment variables
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Replace with your own OpenAI API key or set the OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]

# We first test the LongPaper class for a short text

# We create a shared database for all tests


@pytest.fixture
def local_database():
    obj = LocalDatabase()
    return obj


def test_content_database(local_database):
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext, local_database=local_database)
    long_paper_obj.save_database()
    key_list = local_database.get_list_keys()
    logging.info("Keys in the database: {}".format(key_list))
    assert long_paper_obj.database_id in key_list


def test_long_paper_key(local_database):
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext, local_database=local_database)
    local_key = long_paper_obj.database_id
    assert local_key == "a54d88e06612d820bc3be72877c74f257b561b196f34a3e0e" + \
        "1af181e8a78e70c146682b7ead12846"


def test_long_paper_saving_database(local_database):
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext, local_database=local_database)
    long_paper_obj.save_database()
    assert local_database.check_in_database(long_paper_obj.database_id)


def test_custom_long_paper_key(local_database):
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext, local_database=local_database,
                               database_id="custom_key")
    local_key = long_paper_obj.database_id
    assert local_key == "custom_key"
    long_paper_obj.save_database()
    assert local_database.check_in_database(long_paper_obj.database_id)


def test_long_paper_embedding_caching(local_database):
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext, local_database=local_database)
    embedding = long_paper_obj.calculate_embedding()
    long_paper_obj.save_database()
    embedding = long_paper_obj.calculate_embedding()
    databased_data = local_database.load_from_database(
        long_paper_obj.database_id)
    assert local_database.check_in_database(long_paper_obj.database_id)

    assert "embedding" in databased_data
    assert databased_data['embedding'] == embedding


def test_long_paper_summarizing_caching(local_database):
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext, local_database=local_database)
    summary = long_paper_obj.summarize_longtext_into_chunks()
    long_paper_obj.save_database()
    summary = long_paper_obj.summarize_longtext_into_chunks()
    databased_data = local_database.load_from_database(
        long_paper_obj.database_id)
    assert "summary" in databased_data
    assert databased_data['summary'] == summary


def test_reset_database(local_database):
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext, local_database=local_database)
    long_paper_obj.save_database()
    check_in = local_database.check_in_database(long_paper_obj.database_id)
    assert check_in
    long_paper_obj.reset_database()
    check_in = local_database.check_in_database(long_paper_obj.database_id)
    assert not check_in


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    local_database_fixture = LocalDatabase()
    test_long_paper_key(local_database_fixture)
    test_long_paper_saving_database(local_database_fixture)
    test_custom_long_paper_key(local_database_fixture)
    test_long_paper_embedding_caching(local_database_fixture)
    test_long_paper_summarizing_caching(local_database_fixture)
    test_content_database(local_database_fixture)
    test_reset_database(local_database_fixture)
