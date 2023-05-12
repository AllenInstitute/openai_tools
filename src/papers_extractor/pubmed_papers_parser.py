import requests
from xml.etree import ElementTree as ET

class PubmedPapersParser:
    def __init__(self, query):
        self.query = query
        self.ids = []

    def search_pubmed(self, max_results=1000):
        base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
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
            self.ids.extend([id_elem.text for id_elem in root.findall('IdList/Id')])
        return self.ids
    
    def fetch_details(self):
        base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
        details_per_request = 200  # estimated maximum details per request
        details = []
        for i in range(0, len(self.ids), details_per_request):
            params = {
                'db': 'pubmed',
                'id': ','.join(self.ids[i:i+details_per_request]),
                'retmode': 'xml',
            }
            response = requests.get(base_url, params=params)
            root = ET.fromstring(response.content)
            for article in root.findall('PubmedArticle'):
                detail = self.parse_article(article)
                details.append(detail)
        return details

    def parse_article(self, article):
        title_elem = article.find('MedlineCitation/Article/ArticleTitle')
        abstract_elem = article.find('MedlineCitation/Article/Abstract/AbstractText')
        journal_elem = article.find('MedlineCitation/Article/Journal/Title')
        year_elem = article.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Year')
        author_elems = article.findall('MedlineCitation/Article/AuthorList/Author')
        authors = [f"{elem.find('ForeName').text} {elem.find('LastName').text}" for elem in author_elems]

        detail = {
            'title': title_elem.text if title_elem is not None else None,
            'abstract': abstract_elem.text if abstract_elem is not None else None,
            'journal': journal_elem.text if journal_elem is not None else None,
            'year': year_elem.text if year_elem is not None else None,
            'authors': authors,
        }
        return detail