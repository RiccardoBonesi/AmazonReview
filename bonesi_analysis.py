import IPython as IPython
import pandas as pd
import numpy as np
from collections import namedtuple, Counter
# import tensorflow as tf
from string import punctuation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
# from sklearn.metrics import roc_curve, auc, classification_report
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from nltk import word_tokenize
from sklearn.externals import joblib
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from scipy.sparse import hstack

from gensim import corpora
import pickle
import gensim

import pyLDAvis.gensim
import IPython

import os
java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path




def exploratory_data_analysis(df):
    print("exploratory_data_analysis")
    print("Number of reviews:", len(df))


    # Score Distribution
    ax = plt.axes()
    sns.countplot(df.score, ax=ax)
    ax.set_title('Score Distribution')
    plt.show()

    print("Average Score: ", np.mean(df.score))
    print("Median Score: ", np.median(df.score))


    # creo un nuovo attributo opinion
    # negative = review(1-3), positive=(4-5)
    df.ix[df.score > 3, 'opinion'] = "positive"
    df.ix[df.score <= 3, 'opinion'] = "negative"


    ## opinion distribution
    # ax = plt.axes()
    # sns.countplot(df.opinion, ax=ax)
    # ax.set_title('opinion Positive vs Negative Distribution')
    # plt.show()

    ## userid distribution
    # ax = plt.axes()
    # sns.countplot(df.userid, ax=ax)
    # ax.set_title('UserId Distribution')
    # plt.show()

    ## productid distribution
    # ax = plt.axes()
    # sns.countplot(df.userid, ax=ax)
    # ax.set_title('ProductId Distribution')
    # plt.show()

    print("Proportion of positive review:", len(df[df.opinion == "positive"]) / len(df))
    print("Proportion of positive review:", len(df[df.opinion == "negative"]) / len(df))
    # 77% of the fine food reviews are considered as positive
    # and 23% of them are considered as negative.

    reviews = df.text.values
    labels = df.opinion.values

    ### TEXT REVIEWS ###

    if df.opinion[1] == "positive":
        print("positive" + "\t" + reviews[1][:90] + "...")
    else:
        print("negative" + "\t " + reviews[1][:90] + "...")


    ### Exploratory Visualization ###
    # mi creo un vocabolario di parole positive e negative

    positive_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "positive"]
    negative_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "negative"]

    ### positive
    cnt_positve = Counter()

    for row in positive_reviews:
        cnt_positve.update(row.split(" "))
    print("Vocabulary size for positve reviews:", len(cnt_positve.keys()))

    ### negative
    cnt_negative = Counter()

    for row in negative_reviews:
        cnt_negative.update(row.split(" "))
    print("Vocabulary size for positve reviews:", len(cnt_negative.keys()))

    cnt_total = Counter()

    for row in reviews:
        cnt_total.update(row.split(" "))


    pos_neg_ratio = Counter()
    vocab_pos_neg = (set(cnt_positve.keys())).intersection(set(cnt_negative.keys()))
    for word in vocab_pos_neg:
        if cnt_total[word] > 100:
            ratio = cnt_positve[word] / float(cnt_negative[word] + 1)
            if ratio > 1:
                pos_neg_ratio[word] = np.log(ratio)
            else:
                pos_neg_ratio[word] = -np.log(1 / (ratio + 0.01))

    positive_dict = {}
    for word, cnt in pos_neg_ratio.items():
        if (cnt > 1):
            positive_dict[word] = cnt




    #### positive WORDCLOUD #####
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=positive_dict)

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    ax = plt.axes()
    ax.set_title('Word Cloud with the Highest Positive/Negative Ratio')
    plt.show()

    negative_dict = {}
    for word, cnt in pos_neg_ratio.items():
        if (cnt < 1) & (cnt > 0):
            negative_dict[word] = -np.log(cnt)



    #### negative WORDCLOUD #####
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=negative_dict)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    ax = plt.axes()
    ax.set_title('Word Cloud with the Lowest Positive/Negative Ratio')
    plt.show()

    print("END EXPLORATORY ANALYSIS")


# vocabulary reduction function to reduce
# the vocabulary based on min frequency or polarity.
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


################################
##################################
#################################
###################################





def data_preprocessing(df):

    reviews = df.text.values
    labels = np.array([1 if s == "positive" else 0 for s in df.opinion.values])

    ### remove punctuations
    reviews_cleaned = []
    for i in range(len(reviews)):
        reviews_cleaned.append(''.join([c.lower() for c in reviews[i] if c not in punctuation]))

    # for i in range(len(reviews)):
    #     for c in reviews[i]:
    #         if c not in punctuation:
    #             reviews_cleaned.append(''.join(c.lower()))
    #         else:
    #             reviews_cleaned.append(' '.join)


    print("Before: ", reviews[0])
    print("")
    print("After: ", reviews_cleaned[0])


    ### new vocabulary
    vocabulary = set(' '.join(reviews_cleaned).split())
    print("Vocabulary size: ", len(vocabulary))


    # Vocabulary reduction function to reduce the vocabulary
    # based on min frequency or polarity.
    reviews_cleaned = vocabulary_reduction(reviews_cleaned, labels, min_freq=0, polarity_cut_off=0)


    ### TRANSFORM EACH REVIEW INTO A LIST OF INTEGERS

    # 1) create a dictionary to map each word contained in vocabulary of the reviews to an integer
    # Store all the text from each review in a text variable
    text = ' '.join(reviews_cleaned)

    # List all the vocabulary contained in the reviews
    vocabulary = set(text.split(' '))

    # Map each word to an integer
    vocabulary_to_int = {word: i for i, word in enumerate(vocabulary, 0)}

    reviews_to_int = []
    for i in range(len(reviews_cleaned)):
        to_int = [vocabulary_to_int[word] for word in reviews_cleaned[i].split()]
        reviews_to_int.append(to_int)






    print("END PREPROCESSING")






    ############################
    ##### ASPECT BASED   ########
    #############################
    # ############################


#Selecting only 20 most common aspect.
def get_most_common_aspect(opinion_list):
    opinion = []
    for inner_list in opinion_list:
        for _dict in inner_list:
            for key in _dict:
                opinion.append(key)
    most_common_aspect = [k for k, v in nltk.FreqDist(opinion).most_common(20)]
    return most_common_aspect


#To tag using stanford pos tagger
def posTag(review, stanford_tag):
    tagged_text_list=[]
    for text in review:
        tagged_text_list.append(stanford_tag.tag(word_tokenize(text)))
    return tagged_text_list


def filterTag(tagged_review):
    final_text_list=[]
    for text_list in tagged_review:
        final_text=[]
        for word,tag in text_list:
            if tag in ['NN','NNS','NNP','NNPS','RB','RBR','RBS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ']:
                final_text.append(word)
        final_text_list.append(' '.join(final_text))
    return final_text_list



def get_dict_aspect(y,most_common_aspect):
    position=[]
    for innerlist in y:
        position.append([i for i, j in enumerate(innerlist) if j == 1])
    sorted_common=sorted(most_common_aspect)
    dict_aspect=[]
    for innerlist in position:
        inner_dict={}
        for word in sorted_common:
            if sorted_common.index(word) in innerlist:
                inner_dict[word]= 5
            else:
                inner_dict[word]=0
        dict_aspect.append(inner_dict)
    return dict_aspect



#generate data frame
def get_data_frame(text_list,opinion_list,most_common_aspect):
    data = {'Review': text_list}
    new_df = pd.DataFrame(data)
    for inner_list in opinion_list:
        for _dict in inner_list:
            for key in _dict:
                if key in most_common_aspect:
                    new_df.loc[opinion_list.index(inner_list), key] = _dict[key]
    return new_df


#generate data frame for aspect extraction task
def get_aspect_data_frame(new_df, most_common_aspect):
    for common_aspect in most_common_aspect:
        new_df[common_aspect]=new_df[common_aspect].replace(['positive','negative','neutral','conflict'],[1,1,1,1])
    new_df = new_df.fillna(0)
    return new_df








def aspect_based(df_train, my_df_test):

    ## text_list = text
    ## opinion_list = opinion


    # For stanford POS Tagger
    home = r'Utils\stanford-postagger-full-2017-06-09'

    _path_to_model = home + '/models/english-bidirectional-distsim.tagger'
    _path_to_jar = home + '/stanford-postagger.jar'
    stanford_tag = POS_Tag(model_filename=_path_to_model, path_to_jar=_path_to_jar)


    train_text_list = df_train.text.values
    train_opinion_list = df_train.opinion.values
    most_common_aspect = get_most_common_aspect(train_opinion_list)

    print("start tagging...")

    # This takes time to tag. Already tagged and saved. So, loading file ...
    # tagged_text_list_train=posTag(text_list, stanford_tag)
    # joblib.dump(tagged_text_list_train, 'tagged_text_list_train.pkl')
    tagged_text_list_train = joblib.load('tagged_text_list_train.pkl')

    final_train_text_list = filterTag(tagged_text_list_train)


    df_train = get_data_frame(final_train_text_list, train_opinion_list, most_common_aspect)
    df_train_aspect = get_aspect_data_frame(df_train, most_common_aspect)
    df_train_aspect = df_train_aspect.reindex_axis(sorted(df_train_aspect.columns), axis=1)

    # Similar for test list
    test_text_list = my_df_test.text.values
    test_opinion_list = my_df_test.opinion.values

    tagged_text_list_test=posTag(test_text_list)
    joblib.dump(tagged_text_list_test, 'tagged_text_list_test.pkl')
    # tagged_text_list_test = joblib.load('tagged_text_list_test.pkl')


    final_test_text_list = filterTag(tagged_text_list_test)
    df_test = get_data_frame(final_test_text_list, test_opinion_list, most_common_aspect)
    df_test_aspect = get_aspect_data_frame(df_test, most_common_aspect)
    df_test_aspect = df_test_aspect.reindex_axis(sorted(df_test_aspect.columns), axis=1)


    # Sort the data frame according to aspect's name and separate data(X) and target(y)
    # df_train_aspect = df_train_aspect.sample(frac=1).reset_index(drop=True) #For randoming
    X_train = df_train_aspect.Review
    y_train = df_train_aspect.drop('Review', 1)

    # df_test_aspect = df_test_aspect.sample(frac=1).reset_index(drop=True) #For randoming
    X_test = df_test_aspect.Review
    y_test = df_test_aspect.drop('Review', 1)


    # Change y_train to numpy array
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    # Generate word vecotors using CountVectorizer
    vect = CountVectorizer(max_df=1.0, stop_words='english')
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # Create various models. These are multi-label models.
    nb_classif = OneVsRestClassifier(MultinomialNB()).fit(X_train_dtm, y_train)
    C = 1.0  # SVregularization parameter
    svc = OneVsRestClassifier(svm.SVC(kernel='linear', C=C)).fit(X_train_dtm, y_train)
    lin_svc = OneVsRestClassifier(svm.LinearSVC(C=C)).fit(X_train_dtm, y_train)
    sgd = OneVsRestClassifier(SGDClassifier()).fit(X_train_dtm, y_train)











    print("END ASPECT-BASED")



######################################################
######################################################
######################################################
######################################################


import spacy
# spacy.load('en')
from spacy.lang.en import English
parser = English()

import nltk

# nltk.download('wordnet')
from nltk.corpus import wordnet as wn


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
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens



# import random
# text_data = []
# with open('dataset.csv') as f:
#     for line in f:
#         tokens = prepare_text_for_lda(line)
#         if random.random() > .99:
#             print(tokens)
#             text_data.append(tokens)


def aspect2(df_train, df_test):
    # https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

    reviews = df_train.text.values

    text_data = []
    for r in reviews:
        tokens = prepare_text_for_lda(r)
        text_data.append(tokens)

    # LDA with Gensim
    # First, we are creating a dictionary from the data,
    # then convert to bag-of-words corpus and save the dictionary and corpus for future use.
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')


    # We are asking LDA to find 20 topics in the data
    # NUM_TOPICS = 20
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    # ldamodel.save('model5.gensim')
    # topics = ldamodel.print_topics(num_words=4)
    # for topic in topics:
    #     print(topic)

    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)

    lda10 = gensim.models.ldamodel.LdaModel.load('model5.gensim')
    lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display10)
    pyLDAvis.show(lda_display10)
    pyLDAvis.save_html(lda_display10)






    print("END ASPECT2")




if __name__ == "__main__":
    df = pd.read_csv("Dataset/food.tsv", sep="\t", encoding='latin-1')
    # df = pd.read_csv("cleanedTextCSV.csv", sep="\t", encoding='latin-1')
    # exploratory_data_analysis(df)
    # data_preprocessing(df)
    df_train = df = df.loc[df['productid'] == "B000DZFMEQ"]
    df_test = df[201:250]
    # aspect_based(df_train, df_test)
    aspect2(df_train, df_test)
