from papers_extractor.doi_parser import DoiParser
import logging
import sys


def test_nature_doi_pdf():
    doi_parser = DoiParser('110.1038/nature12373')
    assert doi_parser.get_pdf_link() == \
        'http://www.nature.com/articles/nature12373.pdf'


def test_biorxiv1_doi_pdf():
    doi_parser = DoiParser('10.1101/2020.03.03.972133')
    assert doi_parser.get_pdf_link(
    ) == 'https://www.biorxiv.org/content/10.1101/2020.03.03.972133v1.full.pdf'


def test_biorxiv2_doi_pdf():
    doi_parser = DoiParser('10.1101/2020.10.15.341602')
    assert doi_parser.get_pdf_link(
    ) == 'https://www.biorxiv.org/content/10.1101/2020.10.15.341602v2.full.pdf'


def test_cell_doi_pdf():
    doi_parser = DoiParser('10.1016/j.celrep.2023.112434')
    assert doi_parser.get_pdf_link() is None


def test_title_doi_pdf():
    doi_parser = DoiParser('10.1101/2020.03.03.972133')
    assert doi_parser.get_title(
    ) == ("AI-aided design of novel targeted covalent "
          "inhibitors against SARS-CoV-2")


def test_abstract_doi():
    doi_parser = DoiParser('10.1101/2020.03.03.972133')
    assert doi_parser.get_abstract()[0:10] == '<jats:titl'


def test_author_doi():
    doi_parser = DoiParser('10.1101/2020.03.03.972133')
    assert doi_parser.get_authors()[0]['given'] == 'Bowen'


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_nature_doi_pdf()
    test_biorxiv1_doi_pdf()
    test_biorxiv2_doi_pdf()
    test_cell_doi_pdf()
    test_title_doi_pdf()
    test_abstract_doi()
    test_author_doi()
