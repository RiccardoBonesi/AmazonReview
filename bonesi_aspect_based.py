import pandas as pd
from gensim import corpora
import pickle
import gensim
import pyLDAvis.gensim
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
import absa
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import pandas
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from string import punctuation
from sklearn import svm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import ngrams
from itertools import chain
from wordcloud import WordCloud
from textblob import TextBlob, Word
import matplotlib.pyplot as plt
import seaborn as sns


from spacy.lang.en import English
parser = English()

import nltk
from nltk.util import ngrams


# nltk.download('wordnet')
from nltk.corpus import wordnet as wn


import os
java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path

def label_rate (row):
   if row['score'] == 1 :
      return 'negative'
   if row['score'] == 2 :
      return 'negative'
   if row['score'] == 4 :
      return 'positive'
   if row['score'] == 5 :
      return 'positive'
   return 'neutral'

def plot_rating(data):
    ax = plt.axes()
    sns.countplot(data.score, ax=ax)
    ax.set_title('Score Distribution')
    plt.show()

def word_count(row):
    return len(re.findall(r'\w+', row))
def word_count_log(row):
    return math.log(len(re.findall(r'\w+', row)))

def vocabulary_reduction(reviews, labels, min_freq=10, polarity_cut_off=0.1):
    pos_count = Counter()
    neg_count = Counter()
    tot_count = Counter()

    for i in range(len(reviews)):
        for word in reviews[i].split():
            tot_count[word] += 1
            if labels[i] == 1:
                pos_count[word] += 1
            else:
                neg_count[word] += 1

                # Identify words with frequency greater than min_freq
    vocab_freq = []
    for word in tot_count.keys():
        if tot_count[word] > min_freq:
            vocab_freq.append(word)

            # Use polarity to reduce vocab
    pos_neg_ratio = Counter()
    vocab_pos_neg = (set(pos_count.keys())).intersection(set(neg_count.keys()))
    for word in vocab_pos_neg:
        if tot_count[word] > 100:
            ratio = pos_count[word] / float(neg_count[word] + 1)
            if ratio > 1:
                pos_neg_ratio[word] = np.log(ratio)
            else:
                pos_neg_ratio[word] = -np.log(1 / (ratio + 0.01))

    mean_ratio = np.mean(list(pos_neg_ratio.values()))

    vocab_polarity = []
    for word in pos_neg_ratio.keys():
        if (pos_neg_ratio[word] < (mean_ratio - polarity_cut_off)) or (
                pos_neg_ratio[word] > (mean_ratio + polarity_cut_off)):
            vocab_polarity.append(word)

    vocab_rm_polarity = set(pos_neg_ratio.keys()).difference(vocab_polarity)
    vocab_reduced = (set(vocab_freq)).difference(set(vocab_rm_polarity))

    reviews_cleaned = []

    for review in reviews:
        review_temp = [word for word in review.split() if word in vocab_reduced]
        reviews_cleaned.append(' '.join(review_temp))

    return reviews_cleaned


def data_preprocessing(df):

    reviews = df.text
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # reviews.translate(translator)
    reviews = reviews.apply(lambda row: row.translate(translator))
    reviews = reviews.values
    labels = np.array([1 if s == "positive" else 0 for s in df.PosNeg.values])

    ### remove punctuations
    reviews_cleaned = []
    for i in range(len(reviews)):
        reviews_cleaned.append(''.join([c.lower() for c in reviews[i] if c not in punctuation]))

    print("Before: ", reviews[0])
    print("")
    print("After: ", reviews_cleaned[0])


    ### new vocabulary
    vocabulary = set(' '.join(reviews_cleaned).split())
    print("Vocabulary size: ", len(vocabulary))


    # Vocabulary reduction function to reduce the vocabulary
    # based on min frequency or polarity.
    reviews_cleaned = vocabulary_reduction(reviews_cleaned, labels, min_freq=0, polarity_cut_off=0)


    # ### TRANSFORM EACH REVIEW INTO A LIST OF INTEGERS
    #
    # # 1) create a dictionary to map each word contained in vocabulary of the reviews to an integer
    # # Store all the text from each review in a text variable
    # text = ' '.join(reviews_cleaned)
    #
    # # List all the vocabulary contained in the reviews
    # vocabulary = set(text.split(' '))
    #
    # # Map each word to an integer
    # vocabulary_to_int = {word: i for i, word in enumerate(vocabulary, 0)}
    #
    # reviews_to_int = []
    # for i in range(len(reviews_cleaned)):
    #     to_int = [vocabulary_to_int[word] for word in reviews_cleaned[i].split()]
    #     reviews_to_int.append(to_int)

    print("END PREPROCESSING")
    return reviews_cleaned



def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


from nltk.stem.wordnet import WordNetLemmatizer


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens



# nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))



def prepare_text_for_lda(text):
    tokens = tokenize(text)
    print([token for token in tokens])
    tokens = [token for token in tokens if len(token) > 4]

    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens




def aspect2(df, productId):
    # https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

    reviews = df.cleanedtext.values

    text_data = []
    for r in reviews:
        tokens = prepare_text_for_lda(r)
        print(tokens)
        for doc in tokens:
            tokens[doc] = tokens[doc] + ["_".join(w) for w in ngrams(tokens[doc], 2)]
        text_data.append(tokens)

    # LDA with Gensim
    # First, we are creating a dictionary from the data,
    # then convert to bag-of-words corpus and save the dictionary and corpus for future use.
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')


    # We are asking LDA to find 20 topics in the data
    NUM_TOPICS = 10
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)

    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)

    lda10 = gensim.models.ldamodel.LdaModel.load('model5.gensim')
    lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
    # pyLDAvis.show(lda_display10)
    print("saving LDA...")
    pyLDAvis.save_html(lda_display10, 'LDA/lda_display10' + productId + '.html')
    print("LDA saved: " + productId)



    print("END ASPECT2")




if __name__ == "__main__":

    # df = pd.read_csv("Dataset/food.tsv", sep="\t", encoding='latin-1')
    # df['text'] = [BeautifulSoup(text,"html.parser").get_text() for text in df['text']]
    # unique_product = df.productid.unique()
    # df['PosNeg'] = df.apply(lambda row : label_rate(row), axis=1)
    # # plot_rating(df)
    #
    # df['WordCount'] = df.text.apply(lambda row: word_count(row))
    # # df['WordCountLog'] = df.text.apply(lambda row: word_count_log(row))
    # # print(df.WordCount.min())
    # # plt.hist(df.WordCountLog, bins=np.arange(df.WordCountLog.min(), df.WordCountLog.max() + 1))
    #
    # print(df['userid'].value_counts())
    # df['cleanedtext'] = data_preprocessing(df)
    # # plt.hist(df.userid)
    # # absa.main_sentence(df.cleanedtext[10])
    # # get_most_common_aspect(df.text[1:50])
    # sentiment_scores = list()
    # i = 0
    # for sentence in df.cleanedtext:
    #     line = TextBlob(sentence)
    #     sentiment_scores.append(line.sentiment.polarity)
    #     # print(sentence + ": POLARITY=" + str(line.sentiment.polarity))
    #
    # df['polarity'] = sentiment_scores
    #
    # # sns.distplot(df['polarity'])
    # df.to_csv("cleanedTextCSV.csv", sep='\t', encoding='utf-8')
    df = pd.read_csv("cleanedTextCSV.csv", sep="\t", encoding='latin-1')
    df = df.dropna()

    df1 = df.loc[df['productid'] == "B000DZFMEQ"]
    aspect2(df1, "B000DZFMEQ")
    # df2 = df.loc[df['productid'] == "B00813GRG4"]
    # aspect2(df2, "B00813GRG4")