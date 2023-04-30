# This file contains classes to handle PDF files and extract the publication
# text from them. this will include function to clean up the text and remove
# formatting and irrelevant content.
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import os
import logging
from papers_extractor.openai_parsers import OpenaiLongParser


class PdfParser:
    """This class is used to parse a PDF file and extract the text from it.
    It also has functions to clean up the text and remove formatting and
    irrelevant content.
    """

    def __init__(self, pdf_path, cut_bibliography=True, local_database=None,
                 database_id='auto'):
        """Initializes the class with the path to the PDF file.
        Args:
            pdf_path (str): The path to the PDF file.
            cut_bibliography (bool): Whether to cut the bibliography from the
            text or not. Defaults to True.
            local_database (LocalDatabase): The local database to use. If set
            to None, no database will be used. Defaults to None.
            database_id (str): The key to use for the database. If set to auto,
            it will be generated from the pdf_path. Defaults to auto.
        Returns:
            None
        """

        self.pdf_path = pdf_path
        self.raw_text = None
        self.cleaned_text = None
        self.cut_bibliography = cut_bibliography
        self.database = local_database
        self.database_id = database_id
        self.load_raw_text()

        # We load from the database if requested
        # This will overwrite the raw text if it exists
        if self.database is not None:

            # The key in the database is created from the pdf_path
            if database_id == 'auto':
                self.database_id = pdf_path
            else:
                self.database_id = database_id
            logging.info("Database key for pdf file: {}"
                         .format(self.database_id))

            # We load the database if it exists
            self.database.load_class_from_database(self.database_id, self)

    def load_raw_text(self):
        """Loads the raw text from the PDF file."""

        logging.info("Loading the raw text from the PDF file")
        laparams = LAParams()
        text = extract_text(self.pdf_path, laparams=laparams)
        self.raw_text = text

    def save_database(self):
        """Saves the pdf data to the database if available."""
        if self.database is not None:
            logging.info("Saving database for long paper")
            self.database.save_class_to_database(self.database_id, self)

    def remove_bibliography(self, input_text):
        """We remove the bibliography from the text."""

        updated_text = input_text
        if "References" in self.raw_text:
            updated_text = input_text.split("References")[0]
        if "Bibliography" in self.raw_text:
            updated_text = input_text.split("Bibliography")[0]

        return updated_text

    def get_clean_text(self, chunks_path=None):
        """Extracts the text from the PDF file and cleans it up.
        Args:
            chunks_path (str): The path to the folder where the chunks are
            saved. Defaults to None. Used only for debugging.
        Returns:
            str: The cleaned up text.
        """

        if self.cleaned_text:
            return self.cleaned_text
        else:
            if self.cut_bibliography:
                text_cleaned = self.remove_bibliography(self.raw_text)

            logging.info("Cleaning up and compressing the text")
            openai_prompt = "Clean up formatting, Remove author list, " + \
                "Remove references & bibliography, Remove page number, " + \
                "Remove headers and Remove footers from the following " + \
                "text from a scientific publication. Don't change any " + \
                "other words:"

            AIParser = OpenaiLongParser(text_cleaned, chunk_size=1400)

            if chunks_path is not None:
                if not os.path.exists(chunks_path):
                    os.mkdir(chunks_path)

            all_chunks = AIParser.process_chunks_through_prompt(
                openai_prompt, save_path=chunks_path
            )

            self.cleaned_text = "\n".join(all_chunks)

            self.save_database()

            return self.cleaned_text
