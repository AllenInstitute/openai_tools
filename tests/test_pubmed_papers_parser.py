from papers_extractor.pubmed_papers_parser import PubmedPapersParser
import logging
import sys


def test_search_pubmed():
    # Test that we get some results when we perform a search
    query = PubmedPapersParser('Jerome Lecoq')
    assert len(query.search_pubmed(max_results=3)) > 0


def test_fetch_details():
    # Perform a search and then fetch the details for the first few results
    query = PubmedPapersParser('Jerome Lecoq')

    query.search_pubmed(max_results=3)
    details = query.fetch_details()
    # Test that we get some details
    assert len(details) > 0

    # Test that each paper has a title, and that the title is a string
    for paper in details:
        assert 'title' in paper
        assert isinstance(paper['title'], str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_search_pubmed()
    test_fetch_details()
