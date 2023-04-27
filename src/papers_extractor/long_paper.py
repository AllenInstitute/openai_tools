# This file contains classes to handle long texts that are coming from
# scientific papers. This will include function to make summaries and comments
# of the paper using various deep learning models.
import logging

from papers_extractor.openai_parsers import OpenaiLongParser


class LongPaper:
    """This class is used to summarize a long text."""

    def __init__(self, longtext):
        """Initializes the class with the long text."""
        self.longtext = longtext

    def calculate_embedding(self, chunk_size=3000, parser="GPT"):
        """This function extracts semantic embeddings in chunks"""

        if parser == "GPT":
            local_openai = OpenaiLongParser(self.longtext,
                                            chunk_size=chunk_size)
            self.embedding = local_openai.process_chunks_through_embedding()
            return self.embedding
        else:
            logging.ERROR("Currently only GPT is supported for embedding")
            
    def summarize_longtext_into_chunks(
            self,
            final_chunk_length=2,
            save_path_summary=None,
            max_concurrent_calls=10):
        """This function summarizes a long text into chunks.
        It uses the OpenaiLongParser class to do so.
        """

        openai_prompt = "Write a long, very detailed summary for a \
                technical expert of the following paragraph, from a paper, \
                    refering to the text as -This publication-:"

        current_text = self.longtext

        # we initialize the number of chunks to a large number
        nb_chunks = final_chunk_length + 1

        logging.info("Summarizing the text in chunks")
        while True:
            local_openai = OpenaiLongParser(
                current_text,
                chunk_size=1400,
                max_concurrent_calls=max_concurrent_calls)
            nb_chunks = len(local_openai.chunks)
            if nb_chunks < final_chunk_length:
                break
            logging.info(f"Summarizing chunks:{nb_chunks}")

            summarized_chunks = local_openai.process_chunks_through_prompt(
                openai_prompt, temperature=0, presence_penalty=-0.5
            )
            current_text = "\n".join(summarized_chunks)

        # We can afford to clean up if the text is not too long
        final_long = OpenaiLongParser(
            current_text,
            chunk_size=2000,
            max_concurrent_calls=max_concurrent_calls)
        if final_long.num_chunks == 1:
            logging.info("Cleaning up the summary")

            prompt = "Can you clean up this publication summary to make it \
                flow logically. Keep this summary very technical and detailed:"
            final_text = final_long.process_chunks_through_prompt(
                prompt, temperature=0, presence_penalty=-0.5
            )

        # We save the summary in a txt file
        if save_path_summary:
            with open(save_path_summary, "w") as f:
                f.write("/n".join(final_text))

        return final_text
