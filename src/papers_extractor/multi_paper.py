# In this file, you can find classes to handle a group of long_paper
# object and extract summaries/reviews from them.


class MultiPaper:
    """This class is used to group function that apply to many papers"""

    def __init__(self, longpaper_list):
        """Initializes the class with the long text."""
        self.longpaper_list = longpaper_list
        self.papers_embedding = {}

    def embed_all_papers(self, chunk_size):
        """This is used to extract all paper embedding to make comparisons"""

        # We reset the embedding
        self.papers_embedding = {}

        for indiv_paper in self.longpaper_list:
            paper_embedding = indiv_paper.calculate_embedding(chunk_size)
            self.papers_embedding[indiv_paper.label] = paper_embedding

    def plot_paper_embedding_map(self, plot_tool="t-sne"):
        """This will use t-sne/umap to display embedding maps"""

    def plot_pairwise_distance(self):
        """Using an embedding, this will calculate the pairwise distance
        across all paper combinations. We will calculate the distance between
        papers by measuring the median of all within papers paragraph
        comparisons."""

    def cluster_papers(self, n_cluster, cluster_tool="kmean"):
        """This will cluster group of papers to find semantic close papers"""
        labelled_clusters = ''
        return labelled_clusters
