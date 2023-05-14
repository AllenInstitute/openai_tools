from papers_extractor.multi_paper import MultiPaper
from papers_extractor.unique_paper import UniquePaper
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
    first_paper = UniquePaper("10.1101/2020.03.03.972133")
    second_paper = UniquePaper("10.1016/j.celrep.2023.112434")

    multi_paper = MultiPaper([first_paper, second_paper])
    assert len(multi_paper.papers_list) == 2


def test_multi_paper_embedding():
    first_paper = UniquePaper("10.1101/2020.03.03.972133")
    second_paper = UniquePaper("10.1016/j.celrep.2023.112434")

    multi_paper = MultiPaper([first_paper, second_paper])
    multi_paper.get_embedding_all_papers()

    assert len(multi_paper.papers_embedding) == 2
    assert len(multi_paper.papers_embedding[0][0]) == 1536


def test_multi_paper_plot():
    first_paper = UniquePaper("10.1101/2020.03.03.972133")
    second_paper = UniquePaper("10.1016/j.celrep.2023.112434")

    multi_paper = MultiPaper([first_paper, second_paper])
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
