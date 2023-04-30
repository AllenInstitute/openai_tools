# This file contains classes to handle a little local database
# this is useful to grow a collection of publication and avoid
# duplicated calls to LLM APIs.
# It also stores the embeddings of the papers to avoid recomputing them.
# The intent is to allow changing its storage scheme without having to change
# the rest of the code.
import logging
from diskcache import Cache
import hashlib


def hash_variable(var):
    """This function hashes a variable to use it as a key for the database.
    Args:
        var (str): The variable to hash.
    Returns:
        hashed_var (str): The hashed variable.
    """

    hashed_var = hashlib.sha1(str(var).encode()).hexdigest()
    return hashed_var


class LocalDatabase:
    """This class is used to handle accesses to our local database."""

    def __init__(self, database_path=None):
        """Initializes the class with the long text.
        Args:
            database_path (str): The path to the database. Defaults to None.
            If None, the database will be stored in a temporary folder and will
            be deleted when the program exits.
        Returns:
            None
        """
        logging.info("Opening database")
        if database_path is None:
            self.database = Cache(default_ttl=None)
        else:
            self.database = Cache(database_path, default_ttl=None)

    def __del__(self):
        logging.info("Closing database")
        self.database.close()

    def save_class_to_database(self, key, class_to_save):
        """Saves a class to the database.
        Args:
            key (str): The key to use for the database.
            class_to_save (object): The class to save.
        Returns:
            None
        """
        logging.info("Saving class to database")
        self.database[key] = class_to_save.__dict__

    def load_class_from_database(self, key, class_to_load):
        """Load all properties saved with a class from the database.
        Args:
            key (str): The key to use for the database.
            class_to_load (object): The class to fill with all properties.
        Returns:
            class_to_load (object): The class loaded from the database.
        """
        if self.check_in_database(key) is False:
            logging.info("Key {} not in database".format(key))
        else:
            logging.info("Loading class from database")
            for dict_key in self.database[key]:
                # We do not save the database pointer
                if key != "database":
                    logging.info("Loading {} from database".format(dict_key))
                    class_to_load.__dict__[
                        dict_key] = self.database[key][dict_key]

    def check_in_database(self, key):
        """Checks if a key is in the database.
        Args:
            key (str): The key to use for the database.
        Returns:
            bool: True if the key is in the database, False otherwise.
        """
        logging.info("Checking if {} is in database".format(key))
        return key in self.database

    def save_to_database(self, key, value):
        """Saves a value to the database.
        Args:
            key (str): The key to use for the database.
            value (str): The value to save.
        Returns:
            None
        """
        self.database[key] = value

    def load_from_database(self, key):
        """Loads a value from the database.
        Args:
            key (str): The key to use for the database.
        Returns:
            value (str): The value loaded from the database.
        """
        return self.database[key]

    def reset_key(self, key):
        """Resets data associated with a key."""
        logging.info("Resetting key")
        if key in self.database:
            del self.database[key]

    def get_list_keys(self):
        """Returns the list of keys in the database."""
        keys = list(self.database.iterkeys())
        return keys
