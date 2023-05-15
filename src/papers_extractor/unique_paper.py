# This file contains classes to handle long texts that are coming from
# scientific papers. This will include function to make summaries and comments
# of the paper using various deep learning models.
import logging
from papers_extractor.openai_parsers import OpenaiLongParser
from papers_extractor.arxiv_parser import get_doi_from_arxiv_id
import numpy as np
import re
import datetime
import requests
from bs4 import BeautifulSoup
from habanero import Crossref


# The following methods are used to check if an identifier is a DOI, a PMID or
# an arXiv ID. They are not perfect but should work for most cases.

def check_doi(identifier_string):
    """ Checks if the identifier is a DOI.
    Args:
        identifier_string (str): The identifier to check.
    Returns:
        bool: True if the identifier is a DOI, False otherwise.
    """

    # DOIs start with '10.' and contain a slash
    if identifier_string.startswith('10.') and '/' in identifier_string:
        return True
    return False


def check_pmid(identifier_string):
    """ Checks if the identifier is a PMID.
    Args:
        identifier_string (str): The identifier to check.
    Returns:
        bool: True if the identifier is a PMID, False otherwise.
    """

    # PMIDs are numeric that have increasing numbers depending on when
    # the paper was recorded in PubMed, ie. the first paper has PMID 1
    if identifier_string.isdigit():
        return True
    return False


def check_arxiv(identifier_string):
    """ Checks if the identifier is an arXiv ID.
    Args:
        identifier_string (str): The identifier to check.
    Returns:
        bool: True if the identifier is an arXiv ID, False otherwise.
    """

    # arXiv IDs generally have the format 'category/year.number'
    if re.match(r'[a-z\-]+(\.[a-z\-]+)?/\d{4}\.\d{4,5}', identifier_string):
        return True
    return False


def identify(identifier_string):
    """ Identifies the type of identifier.
    Args:
        identifier_string (str): The identifier to check.
    Returns:
        str: The type of identifier. This can be 'DOI', 'PMID', 'arXiv ID' or
        'Unknown'.
    """
    if check_doi(identifier_string):
        return 'DOI'
    elif check_pmid(identifier_string):
        return 'PMID'
    elif check_arxiv(identifier_string):
        return 'arXiv ID'
    else:
        return 'Unknown'


def check_string_length(string, min_length, max_length):
    """ Checks if the string is of the right length.
    Args:
        string (str): The string to check.
        min_length (int): The minimum length of the string.
        max_length (int): The maximum length of the string.
    Returns:
        bool: True if the string is of the right length, False otherwise.
    """
    if string is None:
        return False
    if len(string) >= min_length and len(string) <= max_length:
        return True
    return False


def get_doi_from_pmid(pmid):
    """This function extracts the PMID from the DOI link."""
    base_url = ("https://eutils.ncbi.nlm.nih.gov/" +
                "entrez/eutils/esearch.fcgi")
    params = {
        "db": "pubmed",
        "term": f"{pmid}[PMID]",
        "retmode": "xml"
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    doi_tag = soup.find("DOI")

    if doi_tag:
        return doi_tag.text
    else:
        logging.warning(f"Could not find DOI for PMID {pmid}")
        return None


def get_pmid_from_doi(doi):
    """This function extracts the PMID from the DOI link."""
    base_url = ("https://eutils.ncbi.nlm.nih.gov/"
                "entrez/eutils/esearch.fcgi")
    params = {
        "db": "pubmed",
        "term": f"{doi}[DOI]",
        "retmode": "xml"
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    pmid_tag = soup.find("Id")

    if pmid_tag:
        pmid = pmid_tag.text
        return pmid
    else:
        pmid = None
        logging.warning(f"Could not find PMID for DOI {doi}")
        return pmid


class UniquePaper:
    """This class is used to hold the information of a given paper.
    """

    def __init__(self, identifier, local_database=None):
        """Initializes a unique paper with the identifier.
        Args:
            identifier (str): The identifier of the paper. This can be a DOI,
            a PMID, an arXiv ID or another identifier that can be used to
            retrieve the paper. If the identifier is a PMID or an arXiv ID,
            the class will try to retrieve the DOI from the identifier.
            local_database (LocalDatabase): The local database to use to
            retrieve the paper. If None, the paper will not be retrieved from
            the database.
        Returns:
            None
        """
        self.identifier = identifier
        identifier_database = identify(identifier)
        if identifier_database == 'DOI':
            self.doi = identifier
            self.pmid = None
            self.arxiv = None
        elif identifier_database == 'PMID':
            # We try to retrieve the DOI from the PMID
            self.doi = get_doi_from_pmid(identifier)
            self.pmid = identifier
            self.arxiv = None
        elif identifier_database == 'arXiv ID':
            self.doi = get_doi_from_arxiv_id(identifier)
            self.pmid = None
            self.arxiv = identifier
        else:
            raise ValueError("The identifier {} is not a recognized \
publication identifier.".format(identifier))

        self.database = local_database

        # These are all the standard fields a paper could have
        # in the context of litterature review
        self.title = None
        self.title_embedding = None
        self.authors = None
        self.abstract = None
        self.abstract_embedding = None
        self.journal = None
        self.year = None
        self.pdf_url = None
        self.pdf_hash = None
        self.fulltext = None
        self.fulltext_embedding = None
        self.longsummary = None
        self.longsummary_embedding = None
        self.nb_citations = None
        self.database_id = None

        # These are fields constructed through API calls
        self.metadata_crossref = None
        self.metadata_pubmed = None

        # This is today's date
        self.last_update = datetime.datetime.now()

        # We attempt to retrieve the paper from the database if available
        if self.database is not None:
            local_id = self.get_database_id()

            logging.debug(f"Database key for this paper: {local_id}")
            self.database.load_class_from_database(local_id, self)

    # We define the following methods to set the fields of the paper
    # This is important to make sure the fields are set correctly
    def set_abstract(self, abstract):
        """Sets the abstract of the paper.
        Args:
            abstract (str): The abstract of the paper.
        Returns:
            None
        """
        if self.abstract is None:
            if not check_string_length(abstract, 10, 10000):
                raise ValueError(
                    (f"The abstract for {self.identifier} is not the " +
                     "right length."))
            self.abstract = abstract
        else:
            logging.debug(f"The abstract for {self.doi} already set.")

    def set_title(self, title):
        """Sets the title of the paper.
        Args:
            title (str): The title of the paper.
        Returns:
            None
        """
        if self.title is None:
            if not check_string_length(title, 10, 1000):
                raise ValueError("The title is not the right length.")
            self.title = title
        else:
            logging.debug(f"The title for {self.doi} already set.")

    def set_authors(self, authors):
        """Sets the authors of the paper.
        Args:
            authors (list): The authors of the paper.
        Returns:
            None
        """
        if self.authors is None:
            if not isinstance(authors, list):
                raise ValueError("The authors are not a list.")
            if len(authors) == 0:
                raise ValueError("The authors list is empty.")
            for author in authors:
                if not check_string_length(author, 5, 100):
                    raise ValueError("An author is not the right length.")
            self.authors = authors
        else:
            logging.debug(f"The authors for {self.doi} already set.")

    def set_journal(self, journal):
        """Sets the journal of the paper.
        Args:
            journal (str): The journal of the paper.
        Returns:
            None
        """
        if self.journal is None:
            if not check_string_length(journal, 2, 200):
                raise ValueError("The journal is not the right length.")
            self.journal = journal
        else:
            logging.debug(f"The journal for {self.doi} already set.")

    def set_year(self, year):
        """Sets the year of the paper.
        Args:
            year (int): The year of the paper.
        Returns:
            None
        """
        if self.year is None:
            if not isinstance(year, int):
                raise ValueError("The year is not an integer.")
            if year < 1900 or year > 2100:
                raise ValueError("The year is not in the right range.")
            self.year = year
        else:
            logging.debug(f"The year for {self.doi} already set.")

    def set_fulltext(self, fulltext):
        """Sets the fulltext of the paper.
        Args:
            fulltext (str): The fulltext of the paper.
        Returns:
            None
        """
        if self.fulltext is None:
            if not check_string_length(fulltext, 100, 1000000):
                raise ValueError("The fulltext is not the right length.")
            self.fulltext = fulltext
        else:
            logging.debug(f"The fulltext for {self.doi} already set.")

    def set_longsummary(self, longsummary):
        """Sets the longsummary of the paper.
        Args:
            longsummary (str): The longsummary of the paper.
        Returns:
            None
        """
        if self.longsummary is None:
            if not check_string_length(longsummary, 100, 1000000):
                raise ValueError("The longsummary is not the right length.")
            self.longsummary = longsummary
        else:
            logging.debug(f"The longsummary for {self.doi} already set.")

    def set_nb_citations(self, nb_citations):
        """Sets the number of citations of the paper.
        Args:
            nb_citations (int): The number of citations of the paper.
        Returns:
            None
        """
        if not isinstance(nb_citations, int):
            raise ValueError("The number of citations is not an integer.")
        if nb_citations < 0:
            raise ValueError("The number of citations is negative.")
        self.nb_citations = nb_citations

    def get_database_id(self):
        """Returns the database id of the paper. We favor the DOI as it is
        the most reliable identifier, followed by the PMID and the arXiv ID.
        Args:
            None
        Returns:
            str: The database id of the paper.
            """
        if self.database_id is not None:
            return self.database_id
        if self.doi is not None:
            self.database_id = self.doi
        elif self.pmid is not None:
            self.database_id = self.pmid
        elif self.arxiv is not None:
            self.database_id = self.arxiv
        else:
            raise ValueError("The paper does not have any identifier.")
        return self.database_id

    def reset_database(self):
        """Resets the database for the long paper if available."""
        if self.database is not None:
            logging.debug("Resetting database for long paper")
            self.database.reset_key(self.database_id)

    def save_database(self):
        """Saves the long paper to the database if available."""
        if self.database is not None:
            logging.debug("Saving long paper to database")
            self.last_update = datetime.datetime.now()
            self.database.save_class_to_database(self.database_id, self)

    def get_label_string(self, format='short'):
        """Returns the label string of the paper. This is a standardized strin
        that contains the title, authors and year of the paper as well as more
        information depending on the requested format.\

        Args:
            format (str): The format of the label string. Can be 'xshort',
            'short', 'medium', 'long', 'xlong'. Defaults to 'short'.
        Returns:
            str: The label string of the paper.
            If the format is 'xshort', the label string will be first author et
            al, publication year.
            If the format is 'short', the label string will be first author et
            al, publication year, journal.
            If the format is 'medium', the label string will be first author et
            al, publication year, journal, title.
            If the format is 'long', the label string will be all authors,
            publication year, journal, title.
            If the format is 'xlong', the label string will be all authors
            , publication year, journal, title, abstract."""

        first_author_lastname = self.get_first_author()
        local_year = self.get_year()

        if self.year is None:
            raise ValueError("The paper does not have any year.")

        # We only create those for the format 'long' and 'xlong'
        if format == 'long' or format == 'xlong':
            # We merge all authors into a single string
            local_authors = self.get_authors()
            local_authors = ', '.join(local_authors)
            if local_authors is None:
                raise ValueError("The paper does not have any author.")

        if format != 'xshort':
            local_journal = self.get_journal()
            if self.journal is None:
                raise ValueError("The paper does not have any journal.")

        if format == 'medium' or format == 'long' or format == 'xlong':
            local_title = self.get_title()
            if self.title is None:
                raise ValueError("The paper does not have any title.")

        if format == 'xlong':
            local_abstract = self.get_abstract()
            if self.abstract is None:
                raise ValueError("The paper does not have any abstract.")

        if format == 'xshort':
            label_string = f"{first_author_lastname} et al., {local_year}"
        elif format == 'short':
            label_string = f"{first_author_lastname} et al., {local_year}, \
{local_journal}"
        elif format == 'medium':
            label_string = f"{first_author_lastname} et al., {local_year} \
{local_journal}\n{local_title}"
        elif format == 'long':
            label_string = f"{local_authors}\n{local_year}, {local_journal}\n\
{local_title}"
        elif format == 'xlong':
            label_string = f"{local_authors}\n{local_year}\n{local_journal}, \
{local_title}\n{local_abstract}"
        else:
            raise ValueError("The format {} is not recognized.".format(format))
        return label_string

    def get_average_embedding(self, field="abstract"):
        """Returns the average embedding of the long paper.
        Args:
            field (str): The field to extract the embeddings from. Can be
            'abstract', 'title', 'fulltext', 'longsummary'. Defaults to
            'abstract'.
        Returns:
            embedding: The averaged embedding of the field.
        """

        local_embedding = self.calculate_embedding(field=field)
        return np.mean(local_embedding, axis=0)

    def calculate_embedding(self, parser="ada2", field="abstract"):
        """This function extracts semantic embeddings in chunks
        from the long text.
        Args:
            parser (str): The parser to use to extract the embeddings.
            Defaults to ada2.
            field (str): The field to extract the embeddings from. Can be
            'abstract', 'title', 'fulltext', 'longsummary'. Defaults to
            'abstract'.
        Returns:
            embedding (list): The list of embeddings for each chunk.
        """
        if field == "abstract":
            local_text = self.get_abstract()
        elif field == "title":
            local_text = self.get_title()
        elif field == "fulltext":
            # These are not available yet
            local_text = None
        elif field == "longsummary":
            # These are not available yet
            local_text = None

        if local_text is None:
            raise ValueError(f"The field {field} is not available for this \
paper.")

        # We check if the embedding of that field is already available
        local_embedding = getattr(self, f"{field}_embedding")
        if local_embedding is not None:
            logging.debug("Embedding already available")
            return local_embedding
        else:
            logging.debug("Calculating embedding for field {}".format(field))
            if parser == "ada2":
                local_openai = OpenaiLongParser(local_text)
                local_embedding, _ = \
                    local_openai.process_chunks_through_embedding()
                setattr(self, f"{field}_embedding", local_embedding)
                self.save_database()
                return local_embedding
            else:
                logging.ERROR("Currently only ada2 is supported for embedding")

    def get_metadata_from_crossref(self):
        if self.metadata_crossref:
            return self.metadata_crossref
        else:
            if self.doi is None:
                logging.warning(
                    f"Could not find crossref metadata for DOI {self.doi}")
                return None
            cr = Crossref()
            try:
                self.metadata_crossref = cr.works(ids=self.doi)
            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTPError for DOI {self.doi}: {e}")
                return None

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

    def get_pmid(self):
        if self.pmid:
            return self.pmid
        else:
            if self.doi is None:
                logging.warning(
                    f"Could not find pubmed metadata for DOI {self.doi}")
                return None
            self.pmid = get_pmid_from_doi(self.doi)
            return self.pmid

    def get_title(self):
        """This function extracts the title from the metadata."""
        if self.title:
            return self.title
        else:
            metadata = self.get_metadata_from_crossref()
            if "message" in metadata and "title" in metadata["message"]:
                title = metadata['message']['title'][0]
            else:
                # We try through the pubmed API
                metadata = self.get_metadata_from_pubmed()
                if metadata and "title" in metadata:
                    title = metadata["title"]
                else:
                    logging.warning(f"Could not find title for DOI {self.doi}")
                    title = None
            self.title = title
            return self.title

    def get_nb_citations(self, force_update=False):
        """This function extracts the citation number from the metadata."""
        if self.nb_citations and not force_update:
            return self.nb_citations
        else:
            metadata = self.get_metadata_from_crossref()
            if (metadata is not None and "message" in metadata
                    and "is-referenced-by-count" in metadata["message"]):
                nb_citations = metadata['message']['is-referenced-by-count']
            else:
                logging.warning(
                    f"Could not find citation number for DOI {self.doi}")
                nb_citations = None
            self.nb_citations = nb_citations
            return self.nb_citations

    def get_abstract(self):
        if self.abstract:
            return self.abstract
        else:
            # Try fetching abstract from CrossRef API
            metadata = self.get_metadata_from_crossref()
            if "message" in metadata and "abstract" in metadata["message"]:
                self.abstract = metadata['message']['abstract']
                return self.abstract
            else:
                logging.warning(
                    (f"Could not find abstract for DOI {self.doi} " +
                     "through CrossRef")
                )
                logging.warning("Trying to fetch abstract from Pubmed API")

            # Try fetching abstract from Pubmed API
            metadata = self.get_metadata_from_pubmed()
            if metadata and "abstract" in metadata:
                self.abstract = metadata["abstract"]
                return self.abstract
            else:
                logging.warning(
                    (f"Could not find abstract for DOI {self.doi} " +
                     "through Pubmed")
                )

            logging.warning(f"abstract not found for DOI {self.doi}")
            self.abstract = None
            return self.abstract

    def get_authors(self):
        """This function extracts the authors from the metadata."""
        if self.authors:
            return self.authors
        else:
            metadata = self.get_metadata_from_crossref()
            if "message" in metadata and "author" in metadata["message"]:
                authors = metadata['message']['author']
                # We merge the given and family names as Crossref separates
                # them
                for index, author in enumerate(authors):
                    author = f"{author['given']} {author['family']}"
                    authors[index] = author
            else:
                # We try through the pubmed API
                metadata = self.get_metadata_from_pubmed()
                if metadata and "authors" in metadata:
                    authors = metadata["authors"]
                else:
                    logging.warning(
                        f"Could not find authors for DOI {self.doi}")
                    authors = None
            self.authors = authors
            return self.authors

    def get_first_author(self):
        """This function extracts the first author last name
        from the metadata."""
        authors = self.get_authors()
        if authors:
            return authors[0].split(" ")[-1]
        else:
            logging.warning(f"Could not find first author for DOI {self.doi}")
            return None

    def get_pdf_url(self):
        """This function extracts the pdf link from the metadata."""
        if self.pdf_url:
            return self.pdf_url
        else:
            metadata = self.get_metadata_from_crossref()

            # If published is biorxiv, then we use the
            # biorxiv api to get the pdf link
            if metadata['message']['publisher'] == ("Cold Spring "
                                                    "Harbor Laboratory"):
                url = 'https://www.biorxiv.org/content/' + self.doi

                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    pdf_link = soup.find(
                        'a', class_='article-dl-pdf-link')['href']
                    self.pdf_url = f"https://www.biorxiv.org{pdf_link}"
                    return self.pdf_url
                else:
                    logging.warning(
                        f"Error: Unable to fetch biorxiv webpage. Status \
                            code {response.status_code}")
                    self.pdf_url = None
                    return self.pdf_url
            else:
                full_text_links = metadata['message'].get('link', [])
                for link in full_text_links:
                    content_type = link.get('content-type', '')
                    if content_type == 'application/pdf':
                        pdf_url = link['URL']
                        self.pdf_url = pdf_url
                        return self.pdf_url
                else:
                    logging.warning(f"No pdf link found for DOI {self.doi}")
                    self.pdf_url = None
                    return self.pdf_url

    def get_year(self):
        """This function extracts the year from the metadata."""
        if self.year:
            return self.year
        else:
            metadata = self.get_metadata_from_crossref()
            if "message" in metadata and "created" in metadata["message"]:
                year = metadata['message']['created']['date-parts'][0][0]
            else:
                # We try through the pubmed API
                metadata = self.get_metadata_from_pubmed()
                if metadata and "pub_date" in metadata:
                    year = metadata["pub_date"].split("-")[0]
                else:
                    logging.warning(f"Could not find year for DOI {self.doi}")
                    year = None
            self.year = year
            return self.year

    def get_journal(self):
        """This function extracts the journal from the metadata."""
        if self.journal is None:
            # We try through the crossref API
            metadata = self.get_metadata_from_crossref()
            if ("message" in metadata
                and "container-title" in metadata["message"]
                    and len(metadata["message"]["container-title"]) > 0):
                journal = metadata['message']['container-title'][0]
            else:
                # We try through the pubmed API
                metadata = self.get_metadata_from_pubmed()
                if metadata and "journal" in metadata:
                    journal = metadata["journal"]
                else:
                    logging.warning(
                        f"Could not find journal for DOI {self.doi}")
                    journal = None

            # We do some cleaning of the journal name
            if "bioRxiv" in journal:
                journal = "bioRxiv"
            self.journal = journal
        return self.journal
