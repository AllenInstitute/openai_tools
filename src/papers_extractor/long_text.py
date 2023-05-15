# This file contains classes to handle long texts that are coming from
# scientific papers. This will include function to make summaries and comments
# of the paper using various deep learning models.
import logging
from papers_extractor.openai_parsers import OpenaiLongParser
from papers_extractor.database_parser import hash_variable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


class LongText:
    """This class is used to process a long text through Large Language Models.
    It will be processed in chunks of a given size if needed."""

    def __init__(self, longtext, chunk_size=1400, local_database=None,
                 database_id='auto'):
        """Initializes the class with the long text.
        Args:
            longtext (str): The long text to summarize.
            chunk_size (int): The size of the chunks in tokens to use for the
            chunks. Defaults to 1400.
            local_database (LocalDatabase): The local database to use.
            If set to None, no database will be used. Defaults to None.
            database_id (str): The key to use for the database. If set to auto,
            it will be generated from the long text and the chunk size.
            We recommend using the DOI of the paper. Defaults
            to auto.
        Returns:
            None
        """
        self.longtext = longtext
        self.chunk_size = chunk_size
        self.database = local_database
        self.summary = None
        self.embedding = None
        self.database_id = database_id

        if self.database is not None:
            # The key in the database is created from the long text and
            # the chunk size unless provided to the class
            if database_id == 'auto':
                self.database_id = hash_variable(self.longtext) + \
                    hash_variable(self.chunk_size)
            else:
                self.database_id = database_id
            logging.debug("Database key for long text: {}"
                          .format(self.database_id))

            self.database.load_class_from_database(self.database_id, self)

    def reset_database(self):
        """Resets the database for the long text if available."""
        if self.database is not None:
            logging.debug("Resetting database for long text")
            self.database.reset_key(self.database_id)

    def save_database(self):
        """Saves the long text to the database if available."""
        if self.database is not None:
            logging.debug("Saving long text to database")
            self.database.save_class_to_database(self.database_id, self)

    def get_average_embedding(self):
        """Returns the average embedding of the long text."""
        if self.embedding is None:
            logging.debug("Embedding not available, calculating it")
            self.calculate_embedding()
        return np.mean(self.embedding, axis=0)

    def plot_tsne_embedding(self, save_figure_path=None,
                            perplexity=5, random_state=42):
        """This function plots the t-SNE embeddings of the long text.
        Args:
            save_figure_path (str): The path to save the figure to. If set to
            None, the figure will not be saved. Defaults to None.
            perplexity (int): The perplexity to use for the t-SNE. Defaults to
            5.
            random_state (int): The random state to use for the t-SNE.
            Defaults to 42.
        Returns:
            fig (matplotlib.pyplot.figure): The figure of the t-SNE plot.
        """

        if self.embedding is None:
            logging.debug("Embedding not available, calculating it")
            self.calculate_embedding()

        matrix = np.array(self.embedding)

        # Create a t-SNE model and transform the data
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    random_state=random_state,
                    init='random',
                    learning_rate=200)
        logging.debug("Fitting t-SNE")
        vis_dims = tsne.fit_transform(matrix)

        x = [x for x, y in vis_dims]
        y = [y for x, y in vis_dims]

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(x, y)
        # Add text next to each dot

        for i, text in enumerate(self.chunks):
            plt.text(x[i] + 1, y[i] + 1, text, fontsize=7)

        plt.title("t-SNE of the embeddings")
        if save_figure_path is not None:
            plt.savefig(save_figure_path)

        return fig

    def calculate_embedding(self, parser="GPT"):
        """This function extracts semantic embeddings in chunks
        from the long text.
        Args:
            parser (str): The parser to use to extract the embeddings.
            Defaults to GPT.
        Returns:
            embedding (list): The list of embeddings for each chunk.
        """

        # We check if the embedding is already available
        if self.embedding is not None:
            return self.embedding
        else:
            logging.debug("Calculating embedding for long text")
            if parser == "GPT":
                local_openai = OpenaiLongParser(self.longtext,
                                                chunk_size=self.chunk_size)
                self.embedding, self.chunks = \
                    local_openai.process_chunks_through_embedding()
                self.save_database()
                return self.embedding
            else:
                logging.ERROR("Currently only GPT is supported for embedding")

    def summarize_longtext_into_chunks(
            self,
            final_chunk_length=2,
            save_path_summary=None,
            max_concurrent_calls=10):
        """This function summarizes a long text into chunks.
        Args:
            final_chunk_length (int): The final number of chunks to have.
            Defaults to 2.
            save_path_summary (str): The path to save the summary.
            Defaults to None.
            max_concurrent_calls (int): The maximum number of concurrent calls
            to the Openai API. Defaults to 10.
        Returns:
            final_text (list): A list of the summary for each chunk.
        """

        openai_prompt = "Write a long, very detailed summary for a " + \
            "technical expert of the following paragraph, from" + \
            " a paper, refering to the text as -This publication-:"

        # We check if the summary is already available
        if self.summary is None:
            current_text = self.longtext

            # we initialize the number of chunks to a large number
            nb_chunks = final_chunk_length + 1

            logging.debug("Summarizing the text in chunks")
            while True:
                local_openai = OpenaiLongParser(
                    current_text,
                    chunk_size=self.chunk_size,
                    max_concurrent_calls=max_concurrent_calls)
                nb_chunks = len(local_openai.chunks)
                if nb_chunks <= final_chunk_length:
                    break
                logging.debug(f"Summarizing chunks:{nb_chunks}")

                summarized_chunks = local_openai.process_chunks_through_prompt(
                    openai_prompt, temperature=0, presence_penalty=-0.5
                )
                current_text = "\n".join(summarized_chunks)

            # This is in case the text is too long to fit in a single chunk
            final_text = current_text

            # We can afford to clean up if the text is not too long
            # Here the chunk size is fixed to maximize the number of tokens
            final_long = OpenaiLongParser(
                current_text,
                chunk_size=2000,
                max_concurrent_calls=max_concurrent_calls)
            if final_long.num_chunks == 1:
                logging.debug("Cleaning up the summary")

                prompt = "Can you clean up this publication summary to " + \
                    "make it flow logically. Keep this summary very " + \
                    "technical and detailed:"
                final_text = final_long.process_chunks_through_prompt(
                    prompt, temperature=0, presence_penalty=-0.5
                )

            # We save the summary in a txt file
            if save_path_summary:
                with open(save_path_summary, "w") as f:
                    f.write("/n".join(final_text))

            self.summary = final_text
            self.save_database()

        return self.summary
