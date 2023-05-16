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
