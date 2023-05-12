# This file contains classes to handle DOI links and extract the
# publication text from them.
from habanero import Crossref
import requests
from bs4 import BeautifulSoup
import logging


class DoiParser:
    """This class is used to handle DOI links and extract the
    publication metadata from them. This includes the title, authors,
    abstract, journal, etc.
    """

    def __init__(self, doi, local_database=None):
        """Initializes the class with the DOI link.
            args:
                doi: This is the DOI link that will be used to extract
                publication text.
                local_database (LocalDatabase): The local database to use.
                If set to None, no database will be used. Defaults to None.
        """
        self.doi = doi
        self.pmid = None
        self.metadata_crossref = None
        self.metadata_pubmed = None
        self.database = local_database

        if self.database is not None:
            self.database.load_class_from_database(self.doi, self)

    def reset_database(self):
        """Resets the database for the doi if available."""
        if self.database is not None:
            logging.info("Resetting database for doi")
            self.database.reset_key(self.doi)

    def save_database(self):
        """Saves the doi data to the database if available."""
        if self.database is not None:
            logging.info("Saving doi to database")
            self.database.save_class_to_database(self.doi, self)

    def get_pmid(self):
        """This function extracts the PMID from the DOI link."""
        if self.pmid:
            return self.pmid
        else:
            base_url = ("https://eutils.ncbi.nlm.nih.gov/"
                        "entrez/eutils/esearch.fcgi")
            params = {
                "db": "pubmed",
                "term": f"{self.doi}[DOI]",
                "retmode": "xml"
            }

            response = requests.get(base_url, params=params)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "xml")
            pmid_tag = soup.find("Id")

            if pmid_tag:
                self.pmid = pmid_tag.text
                return self.pmid
            else:
                self.pmid = 0
                logging.warning(f"Could not find PMID for DOI {self.doi}")
                return self.pmid

    def get_metadata_from_crossref(self):
        if self.metadata_crossref:
            return self.metadata_crossref
        else:
            cr = Crossref()
            self.metadata_crossref = cr.works(ids=self.doi)

            return self.metadata_crossref

    def get_metadata_from_pubmed(self):
        if self.metadata_pubmed:
            return self.metadata_pubmed
        else:
            pmid = self.get_pmid()
            if pmid == 0:
                self.metadata_pubmed = None
                logging.warning(
                    f"Could not find pubmed metadata for DOI {self.doi}")
                return self.metadata_pubmed
            else:
                base_url = ("https://eutils.ncbi.nlm.nih.gov/"
                            "entrez/eutils/efetch.fcgi")
                params = {
                    "db": "pubmed",
                    "retmode": "xml",
                    "id": pmid
                }
                response = requests.get(base_url, params=params)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "xml")
                metadata = {}

                title_tag = soup.find("ArticleTitle")
                if title_tag:
                    metadata["title"] = title_tag.text

                journal_tag = soup.find("Title")
                if journal_tag:
                    metadata["journal"] = journal_tag.text

                authors = []
                for author in soup.find_all("Author"):
                    author_name = (f"{author.find('ForeName').text} \
                                   {author.find('LastName').text}")
                    authors.append(author_name)
                metadata["authors"] = authors

                pub_date_tag = soup.find("PubDate")
                if pub_date_tag:
                    pub_year = pub_date_tag.find("Year")
                    pub_month = pub_date_tag.find("Month")
                    pub_day = pub_date_tag.find("Day")
                    metadata["pub_date"] = (
                        f"{pub_year.text if pub_year else ''}-\
                            {pub_month.text if pub_month else ''}-\
                                {pub_day.text if pub_day else ''}"
                    )

                abstract_tag = soup.find("AbstractText")
                if abstract_tag:
                    metadata["abstract"] = abstract_tag.text

                self.metadata_pubmed = metadata

                return self.metadata_pubmed

    def get_title(self):
        """This function extracts the title from the metadata."""
        metatdata = self.get_metadata_from_crossref()
        if "message" in metatdata and "title" in metatdata["message"]:
            title = metatdata['message']['title'][0]
        else:
            # We try through the pubmed API
            metatdata = self.get_metadata_from_pubmed()
            if metatdata and "title" in metatdata:
                title = metatdata["title"]
            else:
                logging.warning(f"Could not find title for DOI {self.doi}")
                title = None
        return title

    def get_citation_count(self):
        """This function extracts the citation number from the metadata."""
        metatdata = self.get_metadata_from_crossref()
        if ("message" in metatdata 
            and "is-referenced-by-count" in metatdata["message"]):
            citation = metatdata['message']['is-referenced-by-count']
        else:
            logging.warning(
                f"Could not find citation number for DOI {self.doi}")
            citation = None
        return citation

    def get_abstract(self):
        # Try fetching abstract from CrossRef API
        metadata = self.get_metadata_from_crossref()
        if "message" in metadata and "abstract" in metadata["message"]:
            return metadata["message"]["abstract"]
        else:
            logging.warning(
                f"Could not find abstract for DOI {self.doi} through CrossRef")
            logging.warning("Trying to fetch abstract from Pubmed API")

        # Try fetching abstract from Pubmed API
        metadata = self.get_metadata_from_pubmed()
        if metadata and "abstract" in metadata:
            return metadata["abstract"]
        else:
            logging.warning(
                f"Could not find abstract for DOI {self.doi} through Pubmed")

        logging.warning(f"abstract not found for DOI {self.doi}")
        return None

    def get_authors(self):
        """This function extracts the authors from the metadata."""
        metatdata = self.get_metadata_from_crossref()
        if "message" in metatdata and "author" in metatdata["message"]:
            authors = metatdata['message']['author']
        else:
            # We try through the pubmed API
            metatdata = self.get_metadata_from_pubmed()
            if metatdata and "authors" in metatdata:
                authors = metatdata["authors"]
            else:
                logging.warning(f"Could not find authors for DOI {self.doi}")
                authors = None
        return authors

    def get_first_author(self):
        """This function extracts the first author last name
        from the metadata."""
        authors = self.get_authors()
        if authors:
            return authors[0]['family']
        else:
            logging.warning(f"Could not find first author for DOI {self.doi}")
            return None

    def get_pdf_link(self):
        """This function extracts the pdf link from the metadata."""
        metatdata = self.get_metadata_from_crossref()

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
                logging.warning(
                    f"Error: Unable to fetch biorxiv webpage. Status \
                        code {response.status_code}")
                return None
        else:
            full_text_links = metatdata['message'].get('link', [])
            for link in full_text_links:
                content_type = link.get('content-type', '')
                if content_type == 'application/pdf':
                    pdf_url = link['URL']
                    return pdf_url
            else:
                logging.warning(f"No pdf link found for DOI {self.doi}")
                return None

    def get_year(self):
        """This function extracts the year from the metadata."""
        metatdata = self.get_metadata_from_crossref()
        if "message" in metatdata and "created" in metatdata["message"]:
            year = metatdata['message']['created']['date-parts'][0][0]
        else:
            # We try through the pubmed API
            metatdata = self.get_metadata_from_pubmed()
            if metatdata and "pub_date" in metatdata:
                year = metatdata["pub_date"].split("-")[0]
            else:
                logging.warning(f"Could not find year for DOI {self.doi}")
                year = None
        return year

    def get_journal(self):
        """This function extracts the journal from the metadata."""
        metatdata = self.get_metadata_from_crossref()
        if ("message" in metatdata
            and "container-title" in metatdata["message"]
                and len(metatdata["message"]["container-title"]) > 0):
            journal = metatdata['message']['container-title'][0]
        else:
            # We try through the pubmed API
            metatdata = self.get_metadata_from_pubmed()
            if metatdata and "journal" in metatdata:
                journal = metatdata["journal"]
            else:
                logging.warning(f"Could not find journal for DOI {self.doi}")
                journal = None

        # We do some cleaning of the journal name
        if "bioRxiv" in journal:
            journal = "bioRxiv"
        return journal

    def get_citation(self):
        """This returns a standard paper citation in the form
        of first author last name et al. - year - journal"""
        first_author = self.get_first_author()
        year = self.get_year()
        journal = self.get_journal()
        # If we have all the information, we return the citation
        if first_author and year and journal:
            return f"{first_author} et al. - {year} - {journal}"
        else:
            logging.warning(f"Could not find citation for DOI {self.doi}")
            return None
