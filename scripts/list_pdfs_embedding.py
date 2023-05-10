import os
import argparse
# Import the modules from the papers_extractor package
from papers_extractor.pdf_parser import PdfParser
from papers_extractor.long_paper import LongPaper
from papers_extractor.database_parser import LocalDatabase
from papers_extractor.multi_paper import MultiPaper
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
        "--path_folder",
        help="Path to a folder with pdf files",
        type=str,
        default=os.path.join(
            script_path,
            "../../large"),
    )

    parser.add_argument(
        "--database_path",
        help="Path to the database location. This is an optional but \
            very recommended argument. Remember that many calls will be \
            made to summarize your files. \
            If path is not provided, no database will be used or created. \
            If the path is provided, the database will be created if it \
            does not exist. If it exists, it will be loaded and used. \
            Use this to grow your database of papers.",
        type=str,
        default="../../database",
    )

    args = parser.parse_args()

    def get_list_pdf_files(path_folder):
        """This function returns a list of all pdf files in a folder"""
        pdf_files = [
            os.path.join(path_folder, file)
            for file in os.listdir(path_folder)
            if file.endswith(".pdf")
        ]
        return pdf_files

    # We get the path all the pdf files in the folder
    pdf_files = get_list_pdf_files(args.path_folder)

    database_path = args.database_path

    if database_path is not None:
        database_obj = LocalDatabase(database_path=database_path)
    else:
        database_obj = None

    all_legends = []
    all_long_papers = []

    for index, pdf_path in enumerate(pdf_files):
        logging.info("Processing file: {}".format(pdf_path))
        logging.info("File number: {}".format(index))

        # We load the pdf parser to extract and clean the content
        pdf_parser = PdfParser(
            pdf_path,
            cut_bibliography=True,
            local_database=database_obj
        )

        cleaned_text = pdf_parser.get_clean_text()

        # We then use the long paper parser to summarize the content
        paper_parser = LongPaper(cleaned_text, local_database=database_obj)

        all_long_papers.append(paper_parser)

        local_embeddings = paper_parser.calculate_embedding()

        # For now we use the filename as the legend
        filename = os.path.basename(pdf_path)

        all_legends.append(filename.split("-")[0])

    # We create a MultiPaper object to merge all the papers
    multi_paper = MultiPaper(all_long_papers, all_legends)

    # This is the path where the embeddings will be saved
    save_path = os.path.join(args.path_folder, "tsne_embeddings.png")
    logging.info("Saving the t-SNE plot to: {}".format(save_path))

    # We plot the t-SNE plot
    multi_paper.plot_paper_embedding_map(save_path=save_path)
