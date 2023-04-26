# This file contains classes to handle DOI links and extract the
# publication text from them.
from habanero import Crossref
import requests
from bs4 import BeautifulSoup


class DoiParser:
    """This class is used to handle DOI links and extract the
    publication metadata from them.
    """

    def __init__(self, doi):
        """Initializes the class with the DOI link.
            args:
                doi: This is the DOI link that will be used to extract
                publication text.
        """
        self.doi = doi
        self.metadata = None

    def get_doi_metatdata(self):
        if self.metadata:
            return self.metadata
        else:
            cr = Crossref()
            self.metadata = cr.works(ids=self.doi)
            return self.metadata

    def get_title(self):
        """This function extracts the title from the metadata."""
        metatdata = self.get_doi_metatdata()
        title = metatdata['message']['title'][0]
        return title

    def get_abstract(self):
        """This function extracts the abstract from the metadata."""
        metatdata = self.get_doi_metatdata()
        abstract = metatdata['message']['abstract']
        return abstract

    def get_authors(self):
        """This function extracts the authors from the metadata."""
        metatdata = self.get_doi_metatdata()
        authors = metatdata['message']['author']
        return authors

    def get_pdf_link(self):
        """This function extracts the pdf link from the metadata."""
        metatdata = self.get_doi_metatdata()

        # If published is biorxiv, then we use the
        # biorxiv api to get the pdf link
        if metatdata['message']['publisher'] == ("Cold Spring "
                                                 "Harbor Laboratory"):
            url = 'https://www.biorxiv.org/content/' + self.doi

            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                pdf_link = soup.find('a', class_='article-dl-pdf-link')['href']
                return f"https://www.biorxiv.org{pdf_link}"
            else:
                print(
                    f"Error: Unable to fetch biorxiv webpage. Status \
                        code {response.status_code}")
                return None
        else:
            full_text_links = metatdata['message'].get('link', [])
            for link in full_text_links:
                print(link)
                content_type = link.get('content-type', '')
                if content_type == 'application/pdf':
                    pdf_url = link['URL']
                    return pdf_url
            else:
                print("No pdf link found")
                return None
