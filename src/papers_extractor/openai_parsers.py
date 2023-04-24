# This file contains classes to submit texts to the OpenAi API.
# Some functions will enable processing a prompt in chunks to bypass the
# tokens limit.
import logging
import tiktoken
import os
from nltk.tokenize import word_tokenize
import openai
import nltk

nltk.download("punkt")


class OpenaiLongParser:
    """This class is used to submit texts to the OpenAi API.
    Some functions will enable processing a prompt in chunks to bypass the
    tokens limit.
    """

    def __init__(self, longtext, chunk_size=1400):
        """Initializes the class.
        Args:
            longtext (str): The text to submit to the API.
            chunk_size (int): The number of tokens in each chunk.
        """

        self.longtext = longtext
        self.chunk_size = chunk_size
        self.num_tokens = self.count_tokens([longtext])
        self.break_up_longtext_to_chunks(self.longtext, self.chunk_size)
        self.num_chunks = len(self.chunks)

        # We load the API key and send it to OpenAI library
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def count_tokens(self, texts):
        """Counts the number of tokens in the long texts.
        Args:
            texts (list): A list of texts.
        Returns:
            int: The number of tokens.
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
        num_tokens = 0

        for message in texts:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            num_tokens += len(encoding.encode(message))

        return num_tokens

    def custom_word_tokenize(self, text):
        """Tokenizes a string. Currently using a simpler version of the nltk
        word_tokenize function.
        Args:
            text (str): The text to tokenize.
        Returns:
            list: A list of tokens.
        """
        lines = text.splitlines(True)
        tokens = []
        # We tokenize each line separately to keep the line breaks as tokens.
        for line in lines:
            line_tokens = word_tokenize(line.strip())
            tokens.extend(line_tokens)
            tokens.append("\n")
        return tokens[:-1]  # Remove the last newline token

    def break_up_tokens_in_chunks(self, tokens, chunk_size):
        """Breaks up a file into chunks of tokens.
        Args:
            tokens (list): A list of tokens.
            chunk_size (int): The number of tokens in each chunk.
        Returns:
            list: A list of lists of tokens.
        """

        def find_sentence_boundary(tokens, start_idx):
            # This is to only cut at the end of the last full sentence.
            current_length = start_idx

            # We go backward until we find a sentence boundary.
            for idx, token in enumerate(tokens[start_idx::-1]):
                if token in {".", "!", "?"}:
                    current_length = start_idx - idx
                    break
            return current_length

        if len(tokens) <= chunk_size:
            yield tokens
        else:
            end_idx = find_sentence_boundary(tokens, chunk_size)
            chunk = tokens[:end_idx]
            yield chunk
            yield from self.break_up_tokens_in_chunks(tokens[end_idx:],
                                                      chunk_size)

    def convert_to_detokenized_text(self, tokenized_text):
        """Converts a list of tokens to a detokenized string.
        Args:
            tokenized_text (list): A list of tokens.
            Returns:
            str: A detokenized string.
        """

        prompt_text = " ".join(tokenized_text)
        prompt_text = prompt_text.replace(" 's", "'s")
        prompt_text = prompt_text.replace(" ( ", " (")
        prompt_text = prompt_text.replace(" ) ", ") ")
        prompt_text = prompt_text.replace(" , ", ", ")
        prompt_text = prompt_text.replace(" . ", ". ")
        prompt_text = prompt_text.replace(" : ", ": ")
        prompt_text = prompt_text.replace(" ; ", "; ")
        prompt_text = prompt_text.replace(" ! ", "! ")
        prompt_text = prompt_text.replace(" ? ", "? ")
        prompt_text = prompt_text.replace(" % ", "% ")

        return prompt_text

    def break_up_longtext_to_chunks(self, text, chunk_size):
        """Breaks up a file into chunks of tokens.
        Args:
            text (str): The text to break up.
            chunk_size (int): The number of tokens in each chunk.
        Returns:
            list: A list of lists of tokens.
        """
        tokens = self.custom_word_tokenize(text)
        self.chunks = [
            self.convert_to_detokenized_text(local_tokens)
            for local_tokens in self.break_up_tokens_in_chunks(tokens,
                                                               chunk_size)
        ]

    def call_chatGPT(
            self,
            prompt,
            temperature=0.1,
            presence_penalty=0.0,
            frequency_penalty=0.0):
        """Calls the OpenAI API to generate text.
        Args:
            prompt (str): The prompt to use for the API call.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature to use for the API call.
        Returns:
            str: The generated text.
        """

        logging.info("Calling the OpenAI API")
        # Nb of tokens in the prompt

        NbTokensInPrompt = self.count_tokens([prompt])

        logging.info(f"Number of tokens in the prompt: {NbTokensInPrompt}")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000 - NbTokensInPrompt,
            n=1,
            stop=None,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        # Nb to tokens in the response
        NbTokensInResponse = self.count_tokens(
            [response.choices[0].message.content])
        logging.info(f"Number of tokens in the response: {NbTokensInResponse}")

        # Total number of tokens
        TotalNbTokens = NbTokensInPrompt + NbTokensInResponse
        logging.info(f"Total number of tokens: {TotalNbTokens}")

        # We error out if the response was stopped before the end.
        if response.choices[0].finish_reason == "length":
            raise Exception(
                "We stopped because we didn't reach the end of the LLM text."
            )

        return response.choices[0].message.content

    def process_chunks_through_prompt(
        self,
        prompt,
        save_path=None,
        temperature=0.1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ):
        """Processes all the chunks through the API with a given prompt.
        Args:
            prompt (str): The prompt to use for the API call.
            save_path (str): The path to save the chunks to.
        Returns:
            str: The generated text.
        """

        list_chunk = self.chunks
        processed_chunks = []
        logging.info(f"Number of chunks to process: {len(list_chunk)}")

        # We replace this with async calls
        for i, chunk in enumerate(list_chunk):
            logging.info(f"Processing chunk {i}/ {len(list_chunk)}")

            # ChatGPT has a un-tenable desire to finish sentences so we add a .
            # at the end of the prompt
            submit_prompt = prompt + "\n\n" + chunk + "."

            if save_path:
                # We save the input prompt chunk
                chunk_path = os.path.join(save_path, f"input_chunk_{i}.txt")
                with open(chunk_path, "w") as f:
                    f.write(submit_prompt)

            result = self.call_chatGPT(
                submit_prompt,
                temperature=temperature,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            if save_path:
                # We save the output prompt chunk
                chunk_path = os.path.join(save_path, f"output_chunk_{i}.txt")
                with open(chunk_path, "w") as f:
                    f.write(result)

            processed_chunks.append(result)

        return processed_chunks
