# In this file, you can find classes to handle a list of long_text
# objects and extract summaries/reviews from them.
import numpy as np
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import colorsys
from adjustText import adjust_text

# These are helper functions to plot the embedding map


def hsv_to_rgb(h, s, v):
    return np.array([round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v)])


def get_spectrum_colors(N):
    hue_step = 1.0 / N
    colors = [hsv_to_rgb(i * hue_step, 0.75, 1) / 255 for i in range(N)]
    return colors


class MultiPaper:
    """This class is used to group function that apply to many papers"""

    def __init__(self, longpapers_list, labels_list):
        """Initializes the class with many paper objects.
        Args:
            longpapers_list (list): The list of LongPapers to summarize.
            You should use the LongText class to create them and then
            pass them to this class as a list.
            labels_list (list): The list of labels for each long paper. These
            labels will be used to add legends to all plots and analysis. It
            is important that the labels are unique and of the same size as
            the longpapers_list.
            """
        self.longpapers_list = longpapers_list
        self.labels_list = labels_list
        if len(self.longpapers_list) != len(self.labels_list):
            logging.error("The number of long papers and labels "
                          "must be the same")
            assert False

        if len(set(self.labels_list)) != len(self.labels_list):
            logging.error("The labels must be unique")
            assert False

        self.papers_embedding = None

    def get_embedding_all_papers(self):
        """This is used to extract all paper embedding to make comparisons"""
        if not self.papers_embedding:

            # We set the embedding
            self.papers_embedding = {}

            for index, indiv_paper in enumerate(self.longpapers_list):
                paper_embedding = indiv_paper.calculate_embedding()
                local_label = self.labels_list[index]
                self.papers_embedding[local_label] = paper_embedding

        return self.papers_embedding

    def plot_paper_embedding_map(self, save_path=None, perplexity=5, label_proportion=100, list_citations=None):
        """This is used to plot the embedding map of all papers
        Args:
            save_path (str): The path to save the plot
            perplexity (int): The perplexity of the t-SNE
            label_proportion (int): from 0 to 100, the percentage of
            labels to show on the plot. Those are selected randomly.
        """

        dict_embeddings = self.get_embedding_all_papers()

        # We construct the list of all embeddings and legends
        # this is used to plot the t-SNE
        # it is necessary because there can be multiple embeddings
        # for each paper
        all_embeddings = []
        all_legends = []
        all_citations = []
        for local_key in self.labels_list:
            local_embeddings = dict_embeddings[local_key]
            local_legends = [local_key for _ in range(len(local_embeddings))]
            if list_citations is not None:
                citation_nb = list_citations[local_key]
                local_citation = [citation_nb for _ in range(len(local_embeddings))]
                all_citations.extend(local_citation)

            all_embeddings.extend(local_embeddings)
            all_legends.extend(local_legends)

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

        plt.figure(figsize=(width, height))

        # We get the list of unique filenames
        unique_filenames = list(set(all_legends))

        colors = get_spectrum_colors(len(unique_filenames))

        # We create a dictionary to map each filename to a color
        color_dict = {
            filename: color for filename,
            color in zip(
                unique_filenames,
                colors)}

        # We change the background color to black
        plt.gca().set_facecolor('black')

        if list_citations is not None:
            all_citations = np.array(all_citations, dtype=np.float32)
            # We log the citation count for each paper
            all_citations = np.log(all_citations) 
            # We normalize the citation count between 1 and 10
            all_citations = (all_citations - np.min(all_citations)) / (np.max(all_citations) - np.min(all_citations))
            all_citations = all_citations * 15 + 1

            # We convert to int
            all_citations = all_citations.astype(np.int32)


        # We plot each point with the color corresponding to its filename
        if list_citations is None:
            plt.scatter(x, y,
                    c=[color_dict[filename] for filename in all_legends], s=5)
        else:
            plt.scatter(x, y,
                    c=[color_dict[filename] for filename in all_legends], s=all_citations)

        # We add a text on the plot along with each scatter of plot
        # We only print for unique filenames and average the x and y
        # coordinates
        all_texts = []
        x_list_avg = []
        y_list_avg = []
        color_list = []

        for filename in unique_filenames:
            # We draw a random number between 0 and 100
            # If it is below the label_propogation, we add the label
            # to the plot
            # if np.random.randint(0, 100) <= label_proportion:
            # if ('Neuron' in filename) or ("Nat Methods" in filename) or ("Nat Neurosci" in filename):
            # We first get the index of x to average
            list_index = np.where(np.array(all_legends) == filename)[0]

            # If the dot is larger than 10, we add the label
            if np.max(np.array(all_citations)[list_index]) > 7:
                # We then get the average of x and y coordinates
                x_avg = np.median(np.array(x)[list_index])
                y_avg = np.median(np.array(y)[list_index])
                # We then add the text to the plot at the same color as the scatter
                local_text = plt.text(
                    x_avg,  
                    y_avg,
                    filename,
                    fontsize=3,
                    color=color_dict[filename])

                all_texts.append(local_text)
                x_list_avg.append(x_avg)
                y_list_avg.append(y_avg)
                color_list.append(color_dict[filename])

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
