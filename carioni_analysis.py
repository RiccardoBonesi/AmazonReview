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
from stop_words import get_stop_words




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






if __name__  == "__main__":
    # df = pd.read_csv("Dataset/food.tsv",sep = "\t", encoding='latin-1')
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
    # asd = df[df['cleanedtext'].str.contains("nan")]


    df = df.dropna()

    products = df.productid.value_counts()[1:17]

    sentiment_scores = list()
    df = df.loc[df['productid'] == "B000DZFMEQ"]
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]

    i = 0
    for sentence in df.cleanedtext:
        line = TextBlob(sentence)
        sentiment_scores.append(line.sentiment.polarity)
        print(sentence + ": POLARITY=" + str(line.sentiment.polarity))
    comments = TextBlob(' '.join(df.cleanedtext))

    cleaned = list()
    for phrase in comments.noun_phrases:
        count = 0
        for word in phrase.split():
            # Count the number of small words and words without an English definition
            if len(word) <= 2 or (not Word(word).definitions):
                count += 1
        # Only if the 'nonsensical' or short words DO NOT make up more than 40% (arbitrary) of the phrase add
        # it to the cleaned list, effectively pruning the ones not added.
        if count < len(phrase.split()) * 0.4:
            cleaned.append(phrase)

    print("After compactness pruning:\nFeature Size:")
    print(len(cleaned))
    for phrase in cleaned:
        match = list()
        temp = list()
        word_match = list()
        for word in phrase.split():
            # Find common words among all phrases
            word_match = [p for p in cleaned if re.search(word, p) and p not in word_match]
            # If the size of matched phrases set is smaller than 30% of the cleaned phrases,
            # then consider the phrase as non-redundant.
            if len(word_match) <= len(cleaned) * 0.3:
                temp.append(word)
                match += word_match

        phrase = ' '.join(temp)
        #     print("Match for " + phrase + ": " + str(match))

        if len(match) >= len(cleaned) * 0.1:
            # Redundant feature set, since it contains more than 10% of the number of phrases.
            # Prune all matched features.
            for feature in match:
                if feature in cleaned:
                    cleaned.remove(feature)

            # Add largest length phrase as feature
            cleaned.append(max(match, key=len))

    print("After redundancy pruning:\nFeature Size:" + str(len(cleaned)))

    feature_count = dict()
    for phrase in cleaned:
        count = 0
        for word in phrase.split():
            if word not in stopwords.words('english'):
                count += comments.words.count(word)

        print(phrase + ": " + str(count))
        feature_count[phrase] = count

    counts = list(feature_count.values())
    features = list(feature_count.keys())
    threshold = len(comments.noun_phrases) / 20

    print("Threshold:" + str(threshold))

    frequent_features = list()

    for feature, count in feature_count.items():
        if count >= threshold:
            frequent_features.append(feature)

    print('Frequent Features:')
    # sns.set()
    # sns.set_context("poster")
    # f, ax = plt.subplots(figsize=(10, 50))
    # sns.swarmplot(y=features, x=counts, color="c", ax=ax)
    # plt.plot([threshold, threshold], [0, len(features)], linewidth=4, color="r")

    absa_list = dict()
    # For each frequent feature
    for f in frequent_features:
        # For each comment
        absa_list[f] = list()
        for comment in df.cleanedtext:
            blob = TextBlob(comment)
            # For each sentence of the comment
            for sentence in blob.sentences:
                # Search for frequent feature 'f'
                q = '|'.join(f.split())
                if re.search(r'\w*(' + str(q) + ')\w*', str(sentence)):
                    absa_list[f].append(sentence)

    scores = list()
    absa_scores = dict()
    for k, v in absa_list.items():
        absa_scores[k] = list()
        for sent in v:
            score = sent.sentiment.polarity
            scores.append(score)
            absa_scores[k].append(score)

    # Now that we have all the scores, let's plot them!
    # For comparison, we replot the previous global sentiment polarity plot
    # fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(20, 10))
    # plot1 = sns.distplot(scores, ax=ax1)
    #
    # ax1.set_title('Aspect wise scores')
    # ax1.set_xlabel('Sentiment Polarity')
    # ax1.set_ylabel('# of comments')
    #
    # ax2.set_title('Comment wise scores')
    # ax2.set_xlabel('Sentiment Polarity')
    # ax2.set_ylabel('# of comments')
    #
    # plot2 = sns.distplot(sentiment_scores, ax=ax2)

    # Create data values for stripplot and boxplot
    vals = dict()
    vals["aspects"] = list()
    vals["scores"] = list()
    for k, v in absa_scores.items():
        for score in v:
            vals["aspects"].append(k)
            vals["scores"].append(score)

    fig, ax1 = plt.subplots(figsize=(30, 10))

    color = sns.color_palette("Blues", 6)
    plt.xticks(rotation=90)
    sns.set_context("paper", font_scale=3)
    sns.boxplot(x="aspects", y="scores", data=vals, palette=color, ax=ax1)

    print(df)

