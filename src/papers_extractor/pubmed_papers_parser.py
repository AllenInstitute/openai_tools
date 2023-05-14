import requests
from xml.etree import ElementTree as ET
import logging
from papers_extractor.unique_paper import UniquePaper
import time

class PubmedPapersParser:
    def __init__(self, query):
        """This class is used to parse the papers from pubmed.

        Args:
            query (str): The query to search pubmed for.
        """

        self.query = query
        self.ids = []
        self.details = []

    def search_pubmed(self, max_results=1000):
        """This function searches pubmed for the query and returns the ids."""
        if self.ids:
            logging.warning("Ids already fetched, returning cached ids")
            return self.ids
        else:
            logging.info("Searching pubmed for {}".format(self.query))
            base_url = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/e' +
                        'search.fcgi')
            results_per_request = 10000  # maximum results per request
            self.ids = []
            for i in range(0, max_results, results_per_request):
                params = {
                    'db': 'pubmed',
                    'term': self.query,
                    'retmode': 'xml',
                    'retmax': min(results_per_request, max_results - i),
                    'retstart': i,
                }
                response = requests.get(base_url, params=params)
                root = ET.fromstring(response.content)
                self.ids.extend(
                    [id_elem.text for id_elem in root.findall('IdList/Id')])

            # We want to tell the user if there was more than one page of
            # results
            if len(self.ids) == max_results:
                logging.warning("Reached maximum number of results," +
                                "there may be more. You can increase the " +
                                "max_results parameter.")
            logging.info("Found {} papers".format(len(self.ids)))
            return self.ids

    def fetch_details(self):
        """This function fetches the details for the ids."""
        if not self.ids:
            raise Exception("You need to run search_pubmed first")
        if self.details:
            logging.warning(
                "Details already fetched, returning cached details")
            return self.details

        base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
        details_per_request = 200  # estimated maximum details per request
        details = []
        for i in range(0, len(self.ids), details_per_request):
            params = {
                'db': 'pubmed',
                'id': ','.join(self.ids[i:i + details_per_request]),
                'retmode': 'xml',
            }
            response = requests.get(base_url, params=params)
            root = ET.fromstring(response.content)
            for article in root.findall('PubmedArticle'):
                detail = self._parse_article(article)
                details.append(detail)

            time.sleep(1 / 10)

        logging.info("Fetched {} details".format(len(details)))
        self.details = details

        return details

    def get_list_unique_papers(self, local_database=None):
        """This function returns a list of unique paper objects. This is useful
        to call multi paper embedding functions. It also talks to the database
        to check if the paper is already in the database. If it is, it will
        return the object from the database. This is helpful if embeddings
        were already computed. If not, it will create a new
        paper and return it.
        """

        if not self.details:
            raise Exception("You need to run fetch_details first")
        unique_papers = []
        for detail in self.details:
            if detail['doi'] is None:
                identity = detail['pmid']
            else:
                identity = detail['doi']
            try:
                unique_paper = UniquePaper(
                    identity, local_database=local_database)
                unique_paper.set_abstract(detail['abstract'])
                unique_paper.set_title(detail['title'])
                unique_paper.set_year(int(detail['year']))
                unique_paper.set_journal(detail['journal'])
                unique_paper.set_authors(detail['authors'])
                unique_paper.save_database()
                unique_papers.append(unique_paper)
            except ValueError:
                logging.warning(
                    "Could not create unique paper for {}".format(identity))
        return unique_papers

    def _parse_article(self, article):
        title_elem = article.find('MedlineCitation/Article/ArticleTitle')
        abstract_elem = article.find(
            'MedlineCitation/Article/Abstract/AbstractText')
        journal_elem = article.find('MedlineCitation/Article/Journal/Title')
        year_elem = article.find(
            'MedlineCitation/Article/Journal/JournalIssue/PubDate/Year')
        author_elems = article.findall(
            'MedlineCitation/Article/AuthorList/Author')
        doi_elem = article.find(
            'PubmedData/ArticleIdList/ArticleId[@IdType="doi"]')
        pmid = article.find('MedlineCitation/PMID').text
        authors = [
            (
                f"{elem.find('ForeName').text or ''} "
                f"{elem.find('LastName').text or ''}".strip()
            )
            for elem in author_elems
        ]

        detail = {
            'title': title_elem.text if title_elem else None,
            'abstract': abstract_elem.text if abstract_elem else None,
            'journal': journal_elem.text if journal_elem else None,
            'year': year_elem.text if year_elem else None,
            'authors': authors,
            'doi': doi_elem.text if doi_elem else None,
            'pmid': pmid,
        }

        return detail
