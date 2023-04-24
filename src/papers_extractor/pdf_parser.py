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

    def __init__(self, pdf_path, cut_bibliography=True):
        """Initializes the class with the path to the PDF file."""
        self.pdf_path = pdf_path
        self.raw_text = None
        self.cleaned_text = None
        self.cut_bibliography = cut_bibliography

        self.load_saved_texts()

    def load_raw_text(self):
        """Loads the raw text from the PDF file."""

        logging.info("Loading the raw text from the PDF file")
        laparams = LAParams()
        text = extract_text(self.pdf_path, laparams=laparams)
        self.raw_text = text

    def get_raw_text_path(self):
        """Returns the path to the raw text file."""

        return self.pdf_path.replace(".pdf", "_raw.txt")

    def get_cleaned_text_path(self):
        """Returns the path to the cleaned text file."""

        return self.pdf_path.replace(".pdf", "_cleaned.txt")

    def save_raw_text(self):
        """We save the raw text in a txt file."""
        raw_text_path = self.get_raw_text_path()
        with open(raw_text_path, "w") as f:
            f.write(self.raw_text)

    def save_cleaned_text(self):
        """We save the cleaned text in a txt file."""

        cleaned_text_path = self.get_cleaned_text_path()
        with open(cleaned_text_path, "w") as f:
            f.write(self.cleaned_text)

    def load_saved_texts(self):
        """We check if we have already saved the raw text and cleaned text.
        If we have, we load them. If not, we load the raw text and save it.
        """
        raw_text_path = self.get_raw_text_path()
        cleaned_text_path = self.get_cleaned_text_path()
        if os.path.exists(cleaned_text_path):
            with open(cleaned_text_path, "r") as f:
                self.cleaned_text = f.read()
        if os.path.exists(raw_text_path):
            with open(raw_text_path, "r") as f:
                self.raw_text = f.read()
        else:
            self.load_raw_text()
            self.save_raw_text()

    def remove_bibliography(self, input_text):
        """We remove the bibliography from the text."""

        updated_text = input_text
        if "References" in self.raw_text:
            updated_text = input_text.split("References")[0]
        if "Bibliography" in self.raw_text:
            updated_text = input_text.split("Bibliography")[0]

        return updated_text

    def get_clean_text(self):
        """We clean up the text and remove formatting and \
            irrelevant content."""

        if self.cleaned_text:
            return self.cleaned_text
        else:
            if self.cut_bibliography:
                text_cleaned = self.remove_bibliography(self.raw_text)

            logging.info("Cleaning up and compressing the text")
            openai_prompt = "Clean up formatting, Remove author list, \
                Remove references & bibliography, Remove page number, \
                Remove headers and Remove footers from the following \
                text from a scientific publication. Don't change any \
                other words:"

            AIParser = OpenaiLongParser(text_cleaned, chunk_size=1400)

            # We make a folder for the chunks if it doesn't exist
            base_folder = os.path.dirname(self.pdf_path)
            chunks_folder = os.path.join(
                base_folder, os.path.basename(self.pdf_path).split(".pdf")[0]
            )
            if not os.path.exists(chunks_folder):
                os.mkdir(chunks_folder)

            all_chunks = AIParser.process_chunks_through_prompt(
                openai_prompt, save_path=chunks_folder
            )
            self.cleaned_text = "\n".join(all_chunks)

            self.save_cleaned_text()

            return self.save_cleaned_text
