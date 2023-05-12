from papers_extractor.pubmed_papers_parser import PubmedPapersParser
import logging
import sys
import pytest

@pytest.fixture
def papers():
    return PubmedPapersParser('Jerome Lecoq')

def test_search_pubmed(papers):
    # Test that we get some results when we perform a search
    assert len(papers.search_pubmed(max_results=3)) > 0

def test_fetch_details(papers):
    # Perform a search and then fetch the details for the first few results
    papers.search_pubmed(max_results=3)
    details = papers.fetch_details()
    # Test that we get some details
    assert len(details) > 0

    # Test that each paper has a title, and that the title is a string
    for paper in details:
        assert 'title' in paper
        assert isinstance(paper['title'], str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    query = PubmedPapersParser('Jerome Lecoq')
    test_search_pubmed(query)
    test_fetch_details(query)



