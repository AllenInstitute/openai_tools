# In this file, you can find classes to handle a list of long_text
# objects and extract summaries/reviews from them.
import numpy as np
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import colorsys
from adjustText import adjust_text
from papers_extractor.unique_paper import UniquePaper
from matplotlib.lines import Line2D

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
                self.papers_embedding.append(paper_embedding)

        return self.papers_embedding

    def plot_paper_embedding_map(
            self,
            save_path=None,
            perplexity=5,
            field='abstract',
            label='xshort',
            add_citation_count=False
    ):
        """This is used to plot the embedding map of all papers
        Args:
            save_path (str): The path to save the plot
            perplexity (int): The perplexity of the t-SNE
            field (str): The field to use for the embedding. Can be 'abstract'
            'title', 'longsummary' or 'fulltext'
            label (str): The label to use for the plot. Can be 'xshort'
            , 'short', 'medium', 'long' or 'xlong'
        """

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
            if add_citation_count:
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

        width = 12
        height = width / 2  # This will give a 2:1 width-to-height ratio

        # We get the list of unique filenames
        unique_labels = list(set(all_legends))

        colors = get_spectrum_colors(len(unique_labels))

        # We create a dictionary to map each filename to a color
        color_dict = {
            filename: color for filename,
            color in zip(
                unique_labels,
                colors)}

        plt.figure(figsize=(width, height))

        # We change the background color to black
        plt.gca().set_facecolor('black')

        if len(all_citation_count) > 0:
            all_citation_count = np.array(all_citation_count, dtype=np.float32)
            input_citation_count = all_citation_count
            # We log the citation count for each paper
            all_citation_count = np.log(all_citation_count + 1)
            all_citation_count = (all_citation_count
                                  - np.min(all_citation_count)) / \
                (np.max(all_citation_count) - np.min(all_citation_count))
            all_citation_count = all_citation_count * 25 + 1

            # We convert to int
            all_citation_count = all_citation_count.astype(np.int32)
            minimum_citation = np.min(input_citation_count)
            median_citation = np.median(input_citation_count)
            maximum_citation = np.max(input_citation_count)
            minimum_dot_size = np.min(all_citation_count)
            median_dot_size = np.median(all_citation_count)
            maximum_dot_size = np.max(all_citation_count)

        if len(all_citation_count) == 0:
            plt.scatter(x, y, c=[color_dict[local_label]
                        for local_label in all_legends], s=5)
        else:
            plt.scatter(x, y, c=[color_dict[local_label]
                        for local_label in all_legends], s=all_citation_count)
            # We add a legend to the plot to explain the size based on citation
            # count
            local_legend = plt.legend(
                [Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label='Size based on citation count',
                    markerfacecolor='white',
                    markersize=np.sqrt(minimum_dot_size)),
                 Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label='Size based on citation count',
                    markerfacecolor='white',
                    markersize=np.sqrt(median_dot_size)),
                 Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label='Size based on citation count',
                    markerfacecolor='white',
                    markersize=np.sqrt(maximum_dot_size))],
                [f'{int(minimum_citation)} citations',
                    f'{int(median_citation)} citations',
                 f'{int(maximum_citation)} citations'
                 ],
                loc='upper right',
                fontsize=10,
            )
            # Set legend background to transparent
            local_legend.get_frame().set_alpha(0.0)

            # Set legend text color to white
            for text in local_legend.get_texts():
                text.set_color("white")

        # We add a text on the plot along with each scatter of plot
        # We only print for unique filenames and average the x and y
        # coordinates
        all_texts = []
        x_list_avg = []
        y_list_avg = []
        color_list = []

        for local_label in unique_labels:
            # We first get the index of x to average
            list_index = np.where(np.array(all_legends) == local_label)[0]
            # We then get the average of x and y coordinates
            x_avg = np.median(np.array(x)[list_index])
            y_avg = np.median(np.array(y)[list_index])
            # We then add the text to the plot at the same color as the scatter
            local_text = plt.text(
                x_avg,
                y_avg,
                local_label,
                fontsize=5,
                color=color_dict[local_label])

            all_texts.append(local_text)
            x_list_avg.append(x_avg)
            y_list_avg.append(y_avg)
            color_list.append(color_dict[local_label])

        # Adjust the text labels to avoid overlapping
        adjust_text(
            all_texts,
            x=x_list_avg,
            y=y_list_avg,
            arrowprops=dict(
                arrowstyle="-",
                color='w',
                lw=0.5))

        # We adjust the texts to avoid overlapping
        plt.subplots_adjust(bottom=0.1)
        plt.subplots_adjust(right=0.8)

        plt.title("t-SNE of the LLM embeddings of the papers")
        plt.tight_layout()

        # Path to save the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
