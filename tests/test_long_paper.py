from papers_extractor.long_paper import LongPaper
import logging
import sys

# We first test the LongPaper class for a short text


def test_creating_longtext_short_text():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext)

    assert long_paper_obj.longtext == longtext


def test_summarizing_single_chunk_output():
    longtext = "This is a test."
    long_paper_obj = LongPaper(longtext)

    summary = long_paper_obj.summarize_longtext_into_chunks(
        final_chunk_length=1,
        save_path_summary=None,
        max_concurrent_calls=1)

    assert len(summary[0]) > 0
    assert len(summary) == 1


def test_summarizing_double_chunk_output():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext)

    summary = long_paper_obj.summarize_longtext_into_chunks(
        final_chunk_length=2,
        save_path_summary=None,
        max_concurrent_calls=1)

    assert len(summary[0]) > 0
    assert len(summary) == 1


def test_embedding_chunks():
    longtext = "This is a test. It contains two sentences."
    long_paper_obj = LongPaper(longtext, chunk_size=5)

    assert long_paper_obj.longtext == longtext

    embeddings = long_paper_obj.calculate_embedding()

    assert len(embeddings[0]) == 1536
    assert len(embeddings) == 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_creating_longtext_short_text()
    test_summarizing_single_chunk_output()
    test_summarizing_double_chunk_output
    test_embedding_chunks()
