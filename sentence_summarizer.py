from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics import pairwise_distances


def read_article(sentence):
  article = sentence.split(". ")
  sentences = []
  num_sentence = 0
  for sentence in article:
    # print(sentence)
    if len(sentence) < 20 or len(sentence.split(', ')) > 4:
      continue
    sentences.append(sentence.replace("[^a-zA-Z]", " ").lower().split(" "))
    num_sentence += 1
  return num_sentence, sentences


def sentence_similarity(sent1, sent2, stopwords=None):
  if stopwords is None:
    stopwords = []
  # model = SentenceTransformer('bert-base-nli-mean-tokens')
  # sent_emb = model.encode([' '.join(sent1), ' '.join(sent2)])
  # return 1 - cosine_distance(sent_emb[0], sent_emb[1])

  sent1 = [w.lower() for w in sent1]
  sent2 = [w.lower() for w in sent2]

  all_words = list(set(sent1 + sent2))

  vector1 = [0] * len(all_words)
  vector2 = [0] * len(all_words)

  # build the vector for the first sentence
  for w in sent1:
    if w in stopwords:
      continue
    vector1[all_words.index(w)] += 1

  # build the vector for the second sentence
  for w in sent2:
    if w in stopwords:
      continue
    vector2[all_words.index(w)] += 1

  return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
  # Create an empty similarity matrix
  similarity_matrix = np.zeros((len(sentences), len(sentences)))
  # model = SentenceTransformer('bert-base-nli-mean-tokens')
  # sentence_emb = model.encode(sentences)
  # return 1 - pairwise_distances(sentence_emb, metric="cosine")
  for idx1 in tqdm(range(len(sentences))):
    for idx2 in tqdm(range(idx1+1, len(sentences))):
      if idx1 == idx2:  # ignore if both are same sentences
        continue
      similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
  i_upper = np.tril_indices(similarity_matrix.shape[0], -1)
  similarity_matrix[i_upper] = similarity_matrix.T[i_upper]
  return similarity_matrix


def generate_summary(sentences):
  print(sentences)
  stop_words = stopwords.words('english')
  summarize_text = []
  top_n = round(len(sentences.split('.'))**0.5)
  # Step 1 - Read text anc split it
  num_sentence, sentences = read_article(sentences)

  # Step 2 - Generate Similary Martix across sentences
  sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

  # Step 3 - Rank sentences in similarity martix
  sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
  try:
    scores = nx.pagerank(sentence_similarity_graph, max_iter=500)
  except:
    return ''
  # Step 4 - Sort the rank and pick top sentences
  ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
  # print("Indexes of top ranked_sentence order are ", ranked_sentence)

  for i in range(min(top_n, num_sentence)):
    summarize_text.append(" ".join(ranked_sentence[i][1]))

  # Step 5 - Offcourse, output the summarize texr
  print(". ".join(summarize_text))
  return ". ".join(summarize_text)
