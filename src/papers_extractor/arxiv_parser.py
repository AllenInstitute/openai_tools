# In this file, you can find classes to handle arxiv data.
import feedparser

def get_doi_from_arxiv_id(arxiv_id):
    # Query the arXiv API for the given arXiv ID
    url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    feed = feedparser.parse(url)

    # Check the first (and only) entry for a DOI
    if 'arxiv_doi' in feed.entries[0]:
        return feed.entries[0].arxiv_doi
    else:
        return None