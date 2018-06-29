import string
import pandas as pd
import numpy as np
import re
import math
from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


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

def get_most_common_aspect(opinion_list):
    import nltk
    opinion= []
    for inner_list in opinion_list:
        for _dict in inner_list:
            for key in _dict:
                opinion.append(key)
    most_common_aspect = [k for k,v in nltk.FreqDist(opinion).most_common(20)]
    return most_common_aspect

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



def generate_df():
    df = pd.read_csv("Dataset/food.tsv", sep="\t", encoding='latin-1')
    df['text'] = [BeautifulSoup(text,"html.parser").get_text() for text in df['text']]
    unique_product = df.productid.unique()
    df['PosNeg'] = df.apply(lambda row : label_rate(row), axis=1)
    # plot_rating(df)

    df['WordCount'] = df.text.apply(lambda row: word_count(row))
    # df['WordCountLog'] = df.text.apply(lambda row: word_count_log(row))
    # print(df.WordCount.min())
    # plt.hist(df.WordCountLog, bins=np.arange(df.WordCountLog.min(), df.WordCountLog.max() + 1))

    print(df['userid'].value_counts())
    df['cleanedtext'] = data_preprocessing(df)
    # plt.hist(df.userid)
    # absa.main_sentence(df.cleanedtext[10])
    # get_most_common_aspect(df.text[1:50])
    sentiment_scores = list()
    i = 0
    for sentence in df.cleanedtext:
        line = TextBlob(sentence)
        sentiment_scores.append(line.sentiment.polarity)
        # print(sentence + ": POLARITY=" + str(line.sentiment.polarity))

    df['polarity'] = sentiment_scores

    # sns.distplot(df['polarity'])
    df.to_csv("cleanedTextCSV.csv", sep='\t', encoding='utf-8')

    return df