import os
import argparse

# Import the modules from the papers_extractor package
from papers_extractor.pdf_parser import PdfParser
from papers_extractor.long_paper import LongPaper
from papers_extractor.database_parser import LocalDatabase

import logging

# Import the dotenv module to load the environment variables
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Set the logging level to INFO
# This will print the logs in the console
# You can remove this line if you don't want to see the logs
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_pdf",
        help="Path to a pdf file",
        type=str,
        default=os.path.join(
            script_path,
            "../example/2020.12.15.422967v4.full.pdf"),
    )
    parser.add_argument(
        "--save_summary",
        help="Save the summary in a txt file along the pdf file",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--cut_bibliography",
        help="Try not to summarize the bibliography at the end \
            of the pdf file",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--chunk_length",
        help="This is to increase the final length of the summary. The \
            document is summarized in chunks. More chunks means a longer \
            summary. Inconsitencty across the sections could occur with \
            larger number. Typically 1 is a good value for an abstract and \
            2, 3 for more details.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--database_path",
        help="Path to the database file. This is an optional argument. \
            If path is not provided, no database will be used or created. \
            If the path is provided, the database will be created if it \
            does not exist. If it exists, it will be loaded and used. \
            Use this to grow your database of papers.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    pdf_path = args.path_pdf
    save_summary = args.save_summary
    chunk_length = args.chunk_length
    database_path = args.database_path

    if database_path is not None:
        database_obj = LocalDatabase(database_path=database_path)
    else:
        database_obj = None

    # We load the pdf parser to extract and clean the content
    pdf_parser = PdfParser(
        pdf_path,
        cut_bibliography=args.cut_bibliography,
        local_database=database_obj)
    cleaned_text = pdf_parser.get_clean_text()

    # We then use the long paper parser to summarize the content
    paper_parser = LongPaper(cleaned_text, local_database=database_obj)

    # We save the summary in a txt file
    if save_summary:
        summary_path = pdf_path.replace(".pdf", "_summary.txt")
    else:
        summary_path = None

    summary = paper_parser.summarize_longtext_into_chunks(
        final_chunk_length=chunk_length, save_path_summary=summary_path
    )

    # We print the final summary
    print(summary)
