import logging
import os
import sys
import pytest
from papers_extractor.long_paper import LongPaper
from papers_extractor.pdf_parser import PdfParser

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f"{parent_dir}/scripts/")


def do_summarization(max_concurrent_calls):
    """Test out summarization of a pdf file."""
    pdf_path = f"{parent_dir}/example/2021.01.15.426915v3.full.pdf"

    pdf_parser = PdfParser(pdf_path, cut_bibliography=True)

    # We only take the first 1000 characters for the test
    pdf_parser.raw_text = pdf_parser.raw_text[0:1000]

    # We first check if cleaned text exists
    if pdf_parser.cleaned_text is not None:
        pdf_parser.cleaned_text = pdf_parser.cleaned_text[0:1000]

    # We bypass the cleaning of the text for the test
    cleaned_text = pdf_parser.get_clean_text()

    # We then use the long paper parser to summarize the content
    paper_parser = LongPaper(cleaned_text)

    summary = paper_parser.summarize_longtext_into_chunks(
        final_chunk_length=2,
        save_path_summary=None,
        max_concurrent_calls=max_concurrent_calls)
    return summary


@pytest.mark.slow
def test_pdf_summary_sequential():
    """Smoke test for the pdf summary."""
    assert len(do_summarization(1)) > 0


@pytest.mark.slow
def test_pdf_summary_parallel():
    """Smoke test for the pdf summary in parallel."""
    assert len(do_summarization(10)) > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_pdf_summary_parallel()
    test_pdf_summary_sequential()
