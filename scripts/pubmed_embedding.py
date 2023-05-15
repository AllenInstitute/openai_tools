import os
import argparse
# Import the modules from the papers_extractor package
from papers_extractor.database_parser import LocalDatabase
from papers_extractor.multi_paper import MultiPaper
from papers_extractor.pubmed_papers_parser import PubmedPapersParser

import logging

# Import the dotenv module to load the environment variables
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Set the logging level to INFO
# This will print the logs in the console
# You can remove this line if you don't want to see the logs
logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pubmed_query",
        help="Query to search in pubmed",
        type=str,
        default="Jerome Lecoq"
    )

    parser.add_argument(
        "--save_path",
        help="Path to save the plot",
        type=str,
        default="./pubmed_embedding.png",
    )

    parser.add_argument(
        "--field",
        help="Field to use for the embedding, can be title or abstract ",
        type=str,
        default="abstract",
    )

    parser.add_argument(
        "--perplexity",
        help=("Perplexity for the t-SNE plot, default is 8. Higher values "+
              "will make the plot more spread out, lower values will "+
                "make the plot more clustered."),
        type=int,
        default=8,
    )

    parser.add_argument(
        "--add_citation_count",
        help="Whether to add the citation count to the plot",
        type=bool,
        default=False,
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

    QueryObject = PubmedPapersParser(args.pubmed_query)
    QueryObject.search_pubmed()
    QueryObject.fetch_details()

    database_path = args.database_path

    if database_path is not None:
        database_obj = LocalDatabase(database_path=database_path)
    else:
        database_obj = None

    list_unique_papers = QueryObject.get_list_unique_papers(
        local_database=database_obj)

    all_legends = []
    all_long_papers = []

    # We add the citation count to the legend
    """ if args.add_citation_count:
        doi_parser = DoiParser(single_paper["doi"])
        citation_count = doi_parser.get_citation_count()
        legend = "{} - {} citations".format(legend, citation_count) """

    # We create a MultiPaper object to merge all the papers
    multi_paper = MultiPaper(list_unique_papers)

    # This is the path where the embeddings will be saved
    save_path = args.save_path

    logging.info("Saving the t-SNE plot to: {}".format(save_path))

    # We plot the t-SNE plot
    multi_paper.plot_paper_embedding_map(save_path=save_path,
                                         field=args.field,
                                         perplexity=args.perplexity,
                                         label='xshort')
