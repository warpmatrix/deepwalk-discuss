import numpy as np

from GraphEmbedding.ge.classify import read_node_label, Classifier
from GraphEmbedding.ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

edgeListFile = 'GraphEmbedding/data/wiki/Wiki_edgelist.txt'
labelFile = 'GraphEmbedding/data/wiki/wiki_labels.txt'
# edgeListFile = 'GraphEmbedding/data/flight/brazil-airports.edgelist'
# labelFile = 'GraphEmbedding/data/flight/labels-brazil-airports.txt'
# edgeListFile = 'GraphEmbedding/data/flight/europe-airports.edgelist'
# labelFile = 'GraphEmbedding/data/flight/labels-europe-airports.txt'
# edgeListFile = 'GraphEmbedding/data/flight/usa-airports.edgelist'
# labelFile = 'GraphEmbedding/data/flight/labels-usa-airports.txt'

def evaluate_embeddings(embeddings):
    X, Y = read_node_label(labelFile)
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label(labelFile)

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    G = nx.read_edgelist(edgeListFile,
                         create_using=nx.Graph(), nodetype=None, data=[('weight', int)])

    for i in range(0, 10):
        model = DeepWalk(G, walk_length=10, num_walks=80, workers=3)
        model.train(window_size=5, iter=3)
        embeddings = model.get_embeddings()

        evaluate_embeddings(embeddings)
        plot_embeddings(embeddings)
