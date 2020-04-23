from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# papers = pd.read_csv('COVID19Research_summary.csv')
# keywords = ['covid', 'corona']


def get_relevant_papers(papers, keywords):
  """Filter for papers that are relevant to the interest"""
  relevant_papers_doi = []
  for i, abstract in enumerate(papers['abstract']):
    title = papers['title'][i].lower()
    # skip over missing abstract
    if str(abstract) == 'nan':
      continue
    abstract = abstract.lower()
    # check if keywords are in either abstract or title
    if sum([keyword in abstract for keyword in keywords]) + sum([keyword in title for keyword in keywords]) >= 1:
      doi = papers['doi'][i]
      if str(doi) != 'nan':
        relevant_papers_doi.append(doi)
  return relevant_papers_doi


def dump_paper_embedding(papers):
  doi = papers['doi']
  summary = papers['summary']
  # load SBERT
  model = SentenceTransformer('bert-base-nli-mean-tokens')
  abstract = papers['abstract']
  paper2embedding = {}
  for i in tqdm(range(len(doi))):
    sentences = summary.iloc[i]
    # if the summary is null or is too short, then use original abstract to get embedding
    if str(sentences) == 'nan' or len(sentences) < 10:
      sentences = abstract.iloc[i]
    # get overall paragraph embedding from average of sentence embedding
    sentences = np.array(model.encode(sentences.split('. ')))
    paper2embedding[doi.iloc[i]] = np.mean(sentences, axis=0)
  # store sentence embedding
  with open('paper2embedding.pickle', 'wb') as f:
    pickle.dump(paper2embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
  return paper2embedding


def load_paper_embedding():
  """load paper 2 embedding mappings"""
  with open('paper2embedding.pickle', 'rb') as f:
    return pickle.load(f)


def plot_embedding(paper2embedding):
  """plot the embedding"""
  X = np.array([paper2embedding[paper] for paper in list(paper2embedding.keys())])
  pca = PCA(n_components=2)
  result = pca.fit_transform(X)
  plt.scatter(result[:,0], result[:,1])
  plt.show()


def generate_summary(paper2embedding, papers, keywords):
  print("Encoding")
  # get embedding of the papers that are relevant
  print("Get relevant papers")
  relevant_doi = get_relevant_papers(papers, keywords)
  relevant = np.array([paper2embedding[doi] for doi in relevant_doi])
  # get embedding of all papers
  encoded = np.array([paper2embedding[paper] for paper in list(paper2embedding.keys())])
  # get the number of clusters
  print("Get maximal possible clusters")
  n_clusters = maximal_num_cluster(relevant)

  try:
    with open('kmeans'+str(n_clusters)+'.pickle', 'rb') as f:
      kmeans = pickle.load(f)
  except:
    print("Train " + str(n_clusters) + " Means")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(relevant)
    with open('kmeans' + str(n_clusters) + '.pickle', 'wb') as f:
      pickle.dump(kmeans, f, protocol=pickle.HIGHEST_PROTOCOL)

  # # plot embedding and clusters
  # print("Plot Embedding")
  # pca = PCA(n_components=2)
  # result = pca.fit_transform(encoded)
  # centers = pca.transform(kmeans.cluster_centers_)
  # relevant = pca.transform(relevant)
  # plt.scatter(result[:, 0], result[:, 1], label = 'all papers', color='b')
  # plt.scatter(relevant[:, 0], relevant[:, 1], label='relevant papers', color='g')
  # plt.scatter(centers[:, 0], centers[:, 1], label='cluster centers', color='orange')
  # plt.legend()
  # plt.show()
  # print("Make Prediction")
  # find exemplar from embedding space
  avg = []
  for j in range(n_clusters):
      idx = np.where(kmeans.labels_ == j)[0]
      avg.append(np.mean(idx))
  closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, encoded)
  ordering = sorted(range(n_clusters), key=lambda k: avg[k])
  summary_doi = [list(paper2embedding.keys())[closest[idx]] for idx in ordering]+relevant_doi
  summary_abstract = [papers['abstract'][closest[idx]] for idx in ordering]+papers.loc[papers['doi'].isin(relevant_doi)]['abstract'].tolist()
  return summary_doi, summary_abstract


def maximal_num_cluster(X):
  """find maximal reasonable number of cluters by counting number of peaks in a histogram"""
  pca = PCA(n_components=1)
  result = pca.fit_transform(X)
  hist, bin_edge = np.histogram(result, bins=np.linspace(np.min(result), np.max(result), int(len(result)/6)))
  return len(np.where(np.sign(np.diff(hist)))[0])


def papers_summarization(papers, keywords):
  """main function to execute functions in this script"""
  try:
    paper2embedding = load_paper_embedding()
  except:
    paper2embedding = dump_paper_embedding(papers)
  doi, summary = generate_summary(paper2embedding, papers, keywords)
  # print("Summary:", summary)
  return doi, summary
