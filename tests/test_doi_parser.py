from papers_extractor.doi_parser import DoiParser
import logging
import sys
import time
# Below is a list of DOIs, titles and abstracts of papers that will be used
# to test the DoiParser class.
list_doi = [
    '10.1101/2020.03.03.972133',
    '10.1016/j.celrep.2023.112434',
    '10.1038/s41592-021-01285-2',
    '10.1038/nature12373',
    '10.1523/ENEURO.0207-17.2017',
    '10.1016/j.neuron.2022.09.033'
]

list_titles = [
    'AI-aided design of novel targeted covalent inhibitors against SARS-CoV-2',
    'Mitochondrial dynamics define muscle fiber type by modulating ',
    'Removing independent noise in systems neuroscience data using ',
    'Nanometre-scale thermometry in a living cell',
    'Aberrant Cortical Activity in Multiple GCaMP6-Expressing Transgenic',
    'Next-generation brain observatories']

list_abstracts = [
    '<jats:title>Abstract</jats:title><jats:p>The focused drug',
    'Skeletal muscle is highly developed after birth',
    'Progress in many scientific disciplines is hindered by the presence ',
    'Sensitive probing of temperature variations on nanometre scales',
    '<jats:title>Abstract</jats:title><jats:p>Transgenic mouse lines are ',
    'We propose centralized brain observatories for large-scale ']

list_pdf_links = [
    'https://www.biorxiv.org/content/10.1101/2020.03.03.972133v1.full.pdf',
    None,
    'https://www.nature.com/articles/s41592-021-01285-2.pdf',
    'http://www.nature.com/articles/nature12373.pdf',
    None,
    None
]

list_pmids = [
    '32511346',
    '37097817',
    '34650233',
    '23903748',
    '28932809',
    '36240770'
]

list_journal = [
    'bioRxiv',
    'Cell Reports',
    'Nature Methods',
    'Nature',
    'eneuro',
    'Neuron'
]

list_year = [
    2020,
    2023,
    2021,
    2013,
    2017,
    2022
]

list_first_authors = [
    'Tang',
    'Yasuda',
    'Lecoq',
    'Kucsko',
    'Steinmetz',
    'Koch'
]

list_citations = [
    'Tang et al. - 2020 - bioRxiv',
    'Yasuda et al. - 2023 - Cell Reports',
    'Lecoq et al. - 2021 - Nature Methods',
    'Kucsko et al. - 2013 - Nature',
    'Steinmetz et al. - 2017 - eneuro',
    'Koch et al. - 2022 - Neuron'
]


def test_all_doi():
    for i in range(len(list_doi)):
        doi_parser = DoiParser(list_doi[i])

        assert doi_parser.get_title().startswith(list_titles[i])
        assert doi_parser.get_abstract().startswith(list_abstracts[i])
        assert doi_parser.get_pdf_link() == list_pdf_links[i]
        assert doi_parser.get_pmid() == list_pmids[i]
        assert doi_parser.get_journal() == list_journal[i]
        assert doi_parser.get_year() == list_year[i]
        assert doi_parser.get_first_author() == list_first_authors[i]
        assert doi_parser.get_citation() == list_citations[i]
        assert doi_parser.get_citation_count() >= 0
        
        # Sleep for 1/10 second to avoid being blocked by the server
        time.sleep(1 / 10)

def test_citation_count():
    doi_parser = DoiParser('10.1101/2020.03.03.972133')
    assert doi_parser.get_citation_count() > 50

def test_author_doi():
    doi_parser = DoiParser('10.1101/2020.03.03.972133')
    assert doi_parser.get_authors()[0]['given'] == 'Bowen'


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_all_doi()
