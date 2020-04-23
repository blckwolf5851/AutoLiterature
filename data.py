import pandas as pd
import re
from sentence_summarizer import generate_summary
from paper_categorization import papers_summarization


def clean(text):
  text = text.replace('&ndash;', ' - ')
  text = re.sub(r'\([^)]*\)', '', text)
  return text

class Data:
  def __init__(self, filename):
    self.filename = filename
    self.summary_filename = self.filename.replace('.csv', '_summary.csv')
    self.data = pd.read_csv(filename)
    self.data.dropna(subset=['abstract'], inplace=True)
    self.data = self.data[['title', 'doi', 'abstract', 'authors', 'journal', 'has_full_text']]
    self.data['abstract'] = self.data['abstract'].map(clean)

  def generate_summary(self):
    print("Generate Summary")
    self.data['summary'] = self.data['abstract'].map(generate_summary)
    self.store_summary()
    print(self.data)

  def store_summary(self):
    self.data.to_csv(self.summary_filename, index=False)

  def load_summary(self):
    self.data = pd.read_csv(self.summary_filename)

  def extract_paper(self, keywords):
    """select most representative papers that study the same thing as keyword search"""
    try:
      self.load_summary()
    except:
      self.generate_summary()
    print("Generage summary for all paper")
    dois, summaries = papers_summarization(self.data, keywords)
    with open(self.summary_filename.replace('.csv', '.txt'), 'w', encoding='UTF-8') as f:
      seen = []
      for doi, summary in zip(set(dois), set(summaries)):
        if doi in seen:
          continue
        f.write(doi + ',' + summary + '\n')
        seen.append(doi)

if __name__ == '__main__':
  data = Data('COVID19Research.csv')
  data.extract_paper(['covid', 'corona'])


