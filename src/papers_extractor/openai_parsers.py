# This file contains classes to submit texts to the OpenAi API.
# Some functions will enable processing a prompt in chunks to bypass the
# tokens limit.
import asyncio
import logging
import os

import nltk
import openai
import tiktoken
from nltk.tokenize import word_tokenize

nltk.download("punkt")

# Below are methods that can be called outside of the class and
# therefore have a broader scope.


def count_tokens(texts, model="gpt-3.5-turbo-0301"):
    """Counts the number of tokens in the long texts.
    Args:
        texts (list): A list of texts.
        model (str): The model to use.
    Returns:
        int: The number of tokens.
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0

    for message in texts:
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        num_tokens += len(encoding.encode(message))

    return num_tokens


def custom_word_tokenize(text):
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


def custom_word_detokenize(tokenized_text):
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
    prompt_text = prompt_text.replace(" .", ".")
    prompt_text = prompt_text.replace(" : ", ": ")
    prompt_text = prompt_text.replace(" ; ", "; ")
    prompt_text = prompt_text.replace(" ! ", "! ")
    prompt_text = prompt_text.replace(" ? ", "? ")
    prompt_text = prompt_text.replace(" ?", "?")
    prompt_text = prompt_text.replace(" !", "!")
    prompt_text = prompt_text.replace(" % ", "% ")
    prompt_text = prompt_text.replace(" n't", "n't")
    prompt_text = prompt_text.replace("''", "\"")
    prompt_text = prompt_text.replace("``", "\"")
    prompt_text = prompt_text.replace(" 've", "'ve")
    prompt_text = prompt_text.replace(" 'm", "'m")

    return prompt_text

# Below are classes that relates to the OpenAI API.


class OpenaiLongParser:
    """This class is used to submit texts to the OpenAi API.
    Some functions will enable processing a prompt in chunks to bypass the
    tokens limit.
    """

    def __init__(self, longtext, chunk_size=1400, max_concurrent_calls=10):
        """Initializes the class.
        Args:
            longtext (str): The text to submit to the API.
            chunk_size (int): The number of tokens in each chunk.
            max_concurrent_calls (int): The maximum number of concurrent
            calls to the OpenAI API
        """

        self.longtext = longtext
        self.chunk_size = chunk_size
        self.num_tokens = count_tokens([longtext])
        self.break_up_longtext_to_chunks(self.longtext)
        self.num_chunks = len(self.chunks)
        self.max_concurrent_calls = max_concurrent_calls

        # We load the API key and send it to OpenAI library
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def break_up_tokens_in_chunks(self, tokens):
        """Breaks up a file into chunks of tokens.
        Args:
            tokens (list): A list of tokens.
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

        if len(tokens) <= self.chunk_size:
            yield tokens
        else:
            end_idx = find_sentence_boundary(tokens, self.chunk_size)
            chunk = tokens[:end_idx + 1]
            yield chunk
            yield from self.break_up_tokens_in_chunks(tokens[end_idx + 1:])

    def break_up_longtext_to_chunks(self, text):
        """Breaks up a file into chunks of tokens.
        Args:
            text (str): The text to break up.
        Returns:
            list: A list of lists of tokens.
        """
        tokens = custom_word_tokenize(text)
        self.chunks = [
            custom_word_detokenize(local_tokens)
            for local_tokens in self.break_up_tokens_in_chunks(tokens)
        ]

    async def _worker(self, queue):
        """An asynchronous worker that sends the requests to the API."""
        while True:
            (prompt, temperature,
             presence_penalty, frequency_penalty,
             result
             ) = await queue.get()
            NbTokensInPrompt = count_tokens([prompt])
            logging.info("Calling OpenAI API on a chunk of text.")
            logging.info(f"Number of tokens in the prompt: {NbTokensInPrompt}")

            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000 - NbTokensInPrompt,
                n=1,
                stop=None,
                temperature=temperature,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            NbTokensInResponse = count_tokens(
                [response['choices'][0]['message']['content']])
            logging.info(
                f"Number of tokens in the response: {NbTokensInResponse}")

            # Total number of tokens
            TotalNbTokens = NbTokensInPrompt + NbTokensInResponse
            logging.info(f"Total number of tokens: {TotalNbTokens}")

            # We error out if the response was stopped before the end.
            if response.choices[0].finish_reason == "length":
                raise Exception(
                    "We stopped because we didn't reach \
                        the end of the LLM text."
                )

            result.append(response.choices[0]['message']['content'])
            queue.task_done()

    async def async_call_chatGPT(self, prompts, temperature,
                                 presence_penalty, frequency_penalty):
        """Low-level async function that calls chatGPT in parallel."""
        queue = asyncio.Queue()
        workers = [
            asyncio.create_task(
                self._worker(queue)) for _ in range(
                self.max_concurrent_calls)]

        results = [list() for _ in range(len(prompts))]
        for idx, prompt in enumerate(prompts):
            queue.put_nowait(
                (prompt,
                 temperature,
                 presence_penalty,
                 frequency_penalty,
                 results[idx]))

        await queue.join()

        for worker in workers:
            worker.cancel()

        return [x[0] for x in results]

    def multi_call_chatGPT(
            self,
            prompts,
            temperature=0.1,
            presence_penalty=0.0,
            frequency_penalty=0.0):
        """Wrapper that calls the OpenAI API in parallel to generate text.
        Args:
            prompts (List[str]): The prompt to use for the API call.
            temperature (float): The temperature to use for the API call.
            presence penalty (float): The presence penalty to use for
            the API call.
            frequency penalty (float): The frequency penalty to use for
            the API call.
        Returns:
            str: The generated text.
        """
        return asyncio.run(
            self.async_call_chatGPT(
                prompts,
                temperature,
                presence_penalty,
                frequency_penalty))

    def call_embeddingGPT(
            self,
            prompt
    ):
        """Calls the OpenAI API to generate embedding from text.
        Args:
            prompt (str): The prompt to use for the API call.
        Returns:
            array: The embedding.
        """

        logging.info("Calling the OpenAI API")
        # Nb of tokens in the prompt

        NbTokensInPrompt = count_tokens([prompt])

        logging.info(f"Number of tokens in the text: {NbTokensInPrompt}")

        # Below we call openai endpoint for embeddings
        prompt = prompt.replace("\n", " ")

        response = openai.Embedding.create(
            input=[prompt], model="text-embedding-ada-002")

        embedding = response['data'][0]['embedding']

        return embedding

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
            save_path (str): The path to save the chunks to. This is useful
            for debugging.
        Returns:
            str: The generated text.
        """

        list_chunk = self.chunks
        processed_chunks = []
        logging.info(f"Number of chunks to process: {len(list_chunk)}")

        # We replace this with async calls
        submit_prompts = []
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

            submit_prompts.append(submit_prompt)

        processed_chunks = self.multi_call_chatGPT(
            submit_prompts,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        for i, (chunk, result) in enumerate(zip(list_chunk, processed_chunks)):
            if save_path:
                # We save the output prompt chunk
                chunk_path = os.path.join(save_path, f"output_chunk_{i}.txt")
                with open(chunk_path, "w") as f:
                    f.write(result)

        return processed_chunks

    def process_chunks_through_embedding(
        self
    ):
        """Processes all the chunks through the API to extract embeddings.
        Args:
        Returns:
        """

        list_chunk = self.chunks
        processed_chunks = []
        logging.info(f"Number of chunks to embed: {len(list_chunk)}")

        # We replace this with async calls
        for i, chunk in enumerate(list_chunk):
            logging.info(f"Processing chunk {i}/ {len(list_chunk)}")

            # ChatGPT has a un-tenable desire to finish sentences so we
            # add a "." at the end of the prompt
            submit_text = chunk + "."

            result = self.call_embeddingGPT(
                submit_text,
            )

            processed_chunks.append(result)

        return processed_chunks
