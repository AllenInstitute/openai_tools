from papers_extractor.long_paper import LongPaper
import logging
import sys

# We first test the LongPaper class for a short text


def test_summarize_longtext_into_chunks_short_text():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext)

    assert long_paper_obj.longtext == longtext


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_summarize_longtext_into_chunks_short_text()
