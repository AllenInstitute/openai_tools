from papers_extractor.multi_paper import MultiPaper
from papers_extractor.long_paper import LongPaper
import logging
import sys
import os
import openai
import tempfile

# Import the dotenv module to load the environment variables
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Replace with your own OpenAI API key or set the OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]


def test_multi_paper_creation():
    first_long = LongPaper("This is a test")
    second_long = LongPaper("This is a second test")

    labels_list = ["first", "second"]

    multi_paper = MultiPaper([first_long, second_long], labels_list)
    assert len(multi_paper.longpapers_list) == 2

def test_multi_paper_embedding():
    first_long = LongPaper("This is a test")
    second_long = LongPaper("This is a second test")

    labels_list = ["first", "second"]

    multi_paper = MultiPaper([first_long, second_long], labels_list)
    multi_paper.get_embedding_all_papers()
    
    assert len(multi_paper.papers_embedding) == 2
    assert len(multi_paper.papers_embedding["first"][0]) == 1536

def test_multi_paper_plot():
    first_long = LongPaper("This is a test")
    second_long = LongPaper("This is a second test")

    labels_list = ["first", "second"]

    multi_paper = MultiPaper([first_long, second_long], labels_list)
    multi_paper.get_embedding_all_papers()

    # We create a temporary file in pytest tmp folder
    with tempfile.TemporaryDirectory() as tmpdir:
        path_plot = os.path.join(tmpdir, "test_plot.png")
        multi_paper.plot_paper_embedding_map(save_path=path_plot, perplexity=1)

        # We check that the file exists
        assert os.path.exists(path_plot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_multi_paper_creation()
    test_multi_paper_embedding()
    test_multi_paper_plot()