# In this file, you can find classes to handle a list of long_text
# objects and extract summaries/reviews from them.
import numpy as np
import logging
from sklearn.manifold import TSNE
import colorsys
from papers_extractor.unique_paper import UniquePaper
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.models import Scatter
from bokeh.plotting import save
from papers_extractor.openai_parsers import OpenaiLongParser

# These are helper functions to compare papers and plot them


def hsv_to_rgb(h, s, v):
    return np.array([round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v)])


def get_spectrum_colors(N):
    hue_step = 1.0 / N
    colors = [hsv_to_rgb(i * hue_step, 0.75, 1) / 255 for i in range(N)]
    return colors


class MultiPaper:
    """This class is used to group function that apply to many papers"""

    def __init__(self, papers_list):
        """Initializes the class with many paper objects.
        Args:
            papers_list (list): The list of UniquePaper to summarize.
            You should use the UniquePaper class to create them and then
            pass them to this class as a list.
            """

        # We check that the list is not empty
        if not papers_list:
            raise Exception("The list of papers cannot be empty")

        # We check that it is a list of UniquePaper
        if not isinstance(papers_list[0], UniquePaper):
            raise Exception(
                "The list of papers should be a list of UniquePaper")
        self.papers_list = papers_list
        self.papers_embedding = None

    def get_embedding_all_papers(self, field='abstract'):
        """This is used to extract all paper embedding to make comparisons"""
        if not self.papers_embedding:
            # We set the embedding
            self.papers_embedding = []

            for index, indiv_paper in enumerate(self.papers_list):
                paper_embedding = indiv_paper.calculate_embedding(field=field)
                # We save the database after each paper embedding is calculated
                indiv_paper.save_database()
                self.papers_embedding.append(paper_embedding)
                logging.debug(
                    f"Embedding for paper {indiv_paper.identifier} calculated")
                logging.info(("Papers embedding processed: " +
                              f"{index} / {len(self.papers_list)}"))
        return self.papers_embedding

    def plot_paper_embedding_map(
        self,
        save_path=None,
        perplexity=5,
        field='abstract',
        plot_title=None,
    ):
        """This is used to plot the embedding map of all papers
        Args:
            save_path (str): The path to save the plot
            perplexity (int): The perplexity of the t-SNE
            field (str): The field to use for the embedding. Can be 'abstract'
            'title', 'longsummary' or 'fulltext'
            plot_title (str): The title of the plot.
        Returns:
            None
        """

        # We check the arguments

        # We get the paper label
        label = 'short'
        if field not in ['abstract', 'title', 'longsummary', 'fulltext']:
            raise Exception(("field must be 'abstract', 'title', " +
                             "'longsummary' or 'fulltext'"))

        list_embeddings = self.get_embedding_all_papers(field=field)

        # We construct the list of all embeddings and legends
        # this is used to plot the t-SNE
        # it is necessary because there can be multiple embeddings
        # for each paper
        all_embeddings = []
        all_legends = []
        all_citation_count = []
        for index, local_embeddings in enumerate(list_embeddings):
            local_key = self.papers_list[index].get_label_string(
                format=label
            )
            local_legends = [local_key for _ in range(len(local_embeddings))]

            # embeddings can be a list of list
            # so we flatten it
            all_legends.extend(local_legends)
            all_embeddings.extend(local_embeddings)

            local_citation_count = (self.papers_list[index]
                                    .get_nb_citations()
                                    )
            if local_citation_count is None:
                # We set it to 0 if it is None, there is no great way to
                # handle this.
                local_citation_count = 0
            local_citations = [
                local_citation_count for _ in range(
                    len(local_embeddings))]
            all_citation_count.extend(local_citations)
            logging.debug("Got citation count for paper " +
                          "{self.papers_list[index].identifier}")
            # We save the database toi avoid losing the citation count
            self.papers_list[index].save_database()
            logging.info(f"Papers citation processed: {index} / " +
                         f"{len(self.papers_list)}")

        logging.warning(f"Number of included papers: {index}")
        tsne_input_matrix = np.array(all_embeddings)

        # Create a t-SNE model and transform the data
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    random_state=40,
                    init='random',
                    learning_rate=200)

        logging.info("Fitting t-SNE")

        vis_dims = tsne.fit_transform(tsne_input_matrix)

        x = [x for x, y in vis_dims]
        y = [y for x, y in vis_dims]

        width = 1200
        height = int(width / 2)  # This will give a 2:1 width-to-height ratio

        # We get the list of unique filenames
        unique_labels = list(set(all_legends))

        colors = get_spectrum_colors(len(unique_labels))

        # We create a dictionary to map each filename to a color
        color_dict = {
            filename: color for filename,
            color in zip(
                unique_labels,
                colors)}

        all_citation_count = np.array(all_citation_count, dtype=np.float32)
        input_citation_count = all_citation_count
        # We log the citation count for each paper
        all_citation_count = np.log(all_citation_count + 1)
        all_citation_count = (all_citation_count
                              - np.min(all_citation_count)) / \
            (np.max(all_citation_count) - np.min(all_citation_count))
        all_citation_count = all_citation_count * 10 + 1

        # We convert to int
        all_citation_count = all_citation_count.astype(np.int32)
        dot_sizes = all_citation_count

        def rgb_to_hex(rgb):
            return '#%02x%02x%02x' % (
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

        color_dict_hex = {k: rgb_to_hex(v) for k, v in color_dict.items()}

        # Convert the data to a ColumnDataSource - this is a Bokeh object that
        # allows you to map the data to the glyphs
        source = ColumnDataSource(data=dict(
            x=x,
            y=y,
            size=dot_sizes,
            labels=all_legends,
            citations=input_citation_count,
            fill_color=[color_dict_hex[label] for label in all_legends],
        ))

        if plot_title is None:
            plot_title = "t-SNE of the LLM embeddings of the papers"

        p = figure(title=plot_title,
                   x_axis_label='t-sne dim 1',
                   y_axis_label='t-sne dim 2',
                   background_fill_color='black',
                   width=width,  # width of the plot in pixels
                   height=height  # height of the plot in pixels
                   )

        glyph = Scatter(
            x='x',
            y='y',
            size='size',
            fill_color='fill_color',
            marker="circle")
        p.add_glyph(source, glyph)

        # Create the HoverTool with tooltips
        hover = HoverTool(tooltips=[
            ("Paper", "@labels"),
            ("Citations", "@citations"),
        ])

        # Add the HoverTool to the plot
        p.add_tools(hover)

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        # Path to save the plot
        if save_path:
            output_file(save_path)
            save(p)
            show(p)
        else:
            show(p)

    def _process_prompt_through(self, prompt, field):
        """ Process a prompt throught the concatenated abstracts and titles
        of the papers in the list."""
        papers_list = self.papers_list

        if field == 'abstract':
            list_texts = [paper.get_abstract() for paper in papers_list]
        elif field == 'title':
            list_texts = [paper.get_title() for paper in papers_list]
        elif field == 'both':
            list_texts = [paper.get_abstract() + "\n" + paper.get_title()
                          for paper in papers_list]

        input_text = "\n".join(list_texts)

        openai_obj = OpenaiLongParser(
            input_text,
            chunk_size=2000,
            max_concurrent_calls=10)

        prompt = prompt + "\n\n" + input_text + "\n\n"

        final_text = openai_obj.process_chunks_through_prompt(prompt)

        return final_text

    def get_summary_cluster_all_papers(self, field='title'):
        prompt = "Can you summarize in 3-4 words " + \
            "maximum the topic that are " + \
            "shared across the majority of all following papers:"
        result_text = self._process_prompt_through(prompt, field)

        return result_text

    def get_cited_summary_across_all_papers(self):
        single_sentence = self.get_summary_sentence_all_papers(field='title')

        prompt = "Can you write a long accurate scientific abstract " + \
            "without references on the following topic:"

        openai_obj = OpenaiLongParser(
            single_sentence[0],
            chunk_size=2000,
            max_concurrent_calls=10)

        final_text = openai_obj.process_chunks_through_prompt(prompt)
        final_text = "\n".join(final_text)

        list_papers = self.papers_list
        all_citations = []
        real_index_paper = 0
        for index_paper, paper in enumerate(list_papers):
            abstract = paper.get_abstract()
            index_paper = index_paper + 1
            if len(abstract) > 30:
                real_index_paper = real_index_paper + 1

                prompt_citing = "You are writing a scientific review. Here" + \
                    " is an abstract from a paper : \n\n'" \
                    + abstract + "'\n\n" \
                    + "Modify the sentences of the scientific review " + \
                    " to add the insights from the abstract. " + \
                    f"Use the following string '[{real_index_paper}]' when" + \
                    " citing the abstract." + \
                    "Keep all previous citations '[i]' intact." + \
                    "Here is your ongoing scientific review  : \n\n'"

                end_prompt = "I repeat your instructions: Modify the " + \
                    " sentences of the scientific review " + \
                    " to add the insights from the abstract. " + \
                    f"Use the following string '[{real_index_paper}]' " + \
                    "when citing the abstract." + \
                    "Keep all previous citations (e.g. [1] [2], ...) intact."

                openai_obj = OpenaiLongParser(
                    final_text,
                    chunk_size=2000,
                    max_concurrent_calls=10)

                final_text = openai_obj.process_chunks_through_prompt(
                    prompt_citing, temperature=0, end_prompt=end_prompt)
                final_text = "\n".join(final_text)
                local_citation_list = paper.get_label_string('medium')
                all_citations.append(
                    f"[{real_index_paper}] {local_citation_list}")
        # We add citation list
        final_text = final_text + "\n\n" + "\n".join(all_citations)

        return final_text

    def get_summary_sentence_all_papers(self, field='title'):
        """Get the summary of all papers. This function pull the
        title or abstract of all papers and return a summary of them.
        It creates an OpenAi object and uses call to LLM functions with
        a prompt to get the summary. When the summary is too long, it
        is processed in multiple calls."""
        if field not in ['abstract', 'title', 'both']:
            raise Exception(("field must be 'abstract' or 'title'"))

        prompt = "Can you create a sentence that summarize the content of " + \
            "the following papers:\n\n"

        final_text = self._process_prompt_through(prompt, field)

        return final_text
