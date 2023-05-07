import logging
import sys
import os
from papers_extractor.database_parser import hash_file, hash_variable


def test_variable_hashing():
    test_variable = "This is a test"
    local_hash = hash_variable(test_variable)
    assert local_hash == "a54d88e06612d820bc3be72877c74f257b561b19"


def test_file_hashing():
    pdf_file = os.path.join(os.path.dirname(__file__),
                            "../example/2021.01.15.426915v3.full.pdf")
    local_hash = hash_file(pdf_file)
    assert local_hash == "0dc4e899261c6de3b073b8d6e15" + \
        "141f09d70262155d71b2410b35cb621f53a28"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_variable_hashing()
    test_file_hashing()
