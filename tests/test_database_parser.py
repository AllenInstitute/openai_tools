from papers_extractor.long_paper import LongPaper
from papers_extractor.database_parser import LocalDatabase
import logging
import sys


def test_database_start():
    local_database = LocalDatabase()
    assert local_database.database is not None


def test_content_database():
    longtext = "This is a test"
    long_paper_obj = LongPaper(longtext)
    local_database = LocalDatabase()
    local_database.save_class_to_database("test_key", long_paper_obj)
    assert local_database.check_in_database("test_key")
    long_paper_data = local_database.load_from_database("test_key")
    assert long_paper_data["longtext"] == longtext


def test_class_loading_from_database():
    # Here we will ovewrite the content of a local object to
    # test that all properties are correctly loaded from the database
    longtext1 = "This is a first object"
    long_paper_obj1 = LongPaper(longtext1)
    longtext2 = "This is a second object"
    long_paper_obj2 = LongPaper(longtext2)

    local_database = LocalDatabase()
    local_database.save_class_to_database("test_ob1", long_paper_obj1)
    local_database.save_class_to_database("test_ob2", long_paper_obj2)

    assert local_database.check_in_database("test_ob1")
    long_paper_data = local_database.load_from_database("test_ob1")
    assert long_paper_data["longtext"] == longtext1

    assert local_database.check_in_database("test_ob2")
    long_paper_data = local_database.load_from_database("test_ob2")
    assert long_paper_data["longtext"] == longtext2

    # Now we overwrite the content of the object
    local_database.load_class_from_database("test_ob2", long_paper_obj1)
    assert long_paper_obj1.longtext == longtext2


def test_reset_key():
    longtext = "This is a third object"
    long_paper_obj = LongPaper(longtext)
    key = "test_ob3"
    local_database = LocalDatabase()
    local_database.save_class_to_database(key, long_paper_obj)

    assert local_database.check_in_database(key)
    local_database.reset_key(key)
    assert not local_database.check_in_database(key)


def test_list_keys():
    longtext = "This is a fourth object"
    long_paper_obj = LongPaper(longtext)
    key = "test_ob4"
    local_database = LocalDatabase()
    local_database.save_class_to_database(key, long_paper_obj)
    logging.info(local_database.get_list_keys())
    assert key in local_database.get_list_keys()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    test_database_start()
    test_content_database()
    test_class_loading_from_database()
    test_reset_key()
    test_list_keys()
