from papers_extractor.pdf_parser import PdfParser
import os

# We first test that the pdf parser can read one of the example pdf file
# and extract the text from it.


def test_read_pdf():
    example_dir = os.path.join(os.path.dirname(__file__), '..', 'example')
    example_pdf = os.path.join(example_dir, '2020.12.15.422967v4.full.pdf')

    pdf_parser = PdfParser(example_pdf)
    pdf_parser.load_raw_text()

    print(pdf_parser.raw_text[0:9])
    assert pdf_parser.raw_text[0:16] == 'bioRxiv preprint'
