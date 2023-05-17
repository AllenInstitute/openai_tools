from papers_extractor.multi_paper import MultiPaper
from papers_extractor.unique_paper import UniquePaper
import logging
import sys
import os
import openai
from papers_extractor.pubmed_papers_parser import PubmedPapersParser

# Import the dotenv module to load the environment variables
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Replace with your own OpenAI API key or set the OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]


def test_pubmed_query_summary():
    query = PubmedPapersParser('Jerome Lecoq')

    query.search_pubmed(max_results=3)
    query.fetch_details()
    list_papers = query.get_list_unique_papers()
    multi_paper = MultiPaper(list_papers)
    summary = multi_paper.get_summary_sentence_all_papers(field='title')
    assert len(summary[0]) > 2

def test_multi_paper_summary():
    first_paper = UniquePaper("10.1101/2020.03.03.972133")
    second_paper = UniquePaper("10.1016/j.celrep.2023.112434")

    multi_paper = MultiPaper([first_paper, second_paper])
    summary = multi_paper.get_summary_sentence_all_papers(field='title')
    assert len(summary[0]) > 2

def test_multi_paper_cluster():
    query = PubmedPapersParser('Mark Schnitzer')

    query.search_pubmed(max_results=3)
    query.fetch_details()
    list_papers = query.get_list_unique_papers()
    multi_paper = MultiPaper(list_papers)
    summary = multi_paper.get_summary_cluster_all_papers(field='title')
    assert len(summary[0]) > 2

def test_closed_semantic():
    query = PubmedPapersParser('tasic bosiljka transcriptomic visual taxonomy')

    query.search_pubmed(max_results=2)
    query.fetch_details()
    list_papers = query.get_list_unique_papers()

    multi_paper = MultiPaper(list_papers)
    summary = multi_paper.get_cited_summary_across_all_papers()
    assert len(summary[0]) > 2

def test_cited_summary():
    query = PubmedPapersParser('Jerome Lecoq')

    query.search_pubmed(max_results=2)
    query.fetch_details()
    list_papers = query.get_list_unique_papers()

    multi_paper = MultiPaper(list_papers)
    summary = multi_paper.get_cited_summary_across_all_papers()
    assert len(summary[0]) > 2

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_multi_paper_summary()
    test_pubmed_query_summary()
    test_multi_paper_cluster()
    test_cited_summary()
    test_closed_semantic()