from dataset_utils import *

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis.gensim
import seaborn as sns
from gensim import corpora
from spacy.lang.en import English
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib
import pickle
import warnings
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import os
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

parser = English()

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path



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
    # print([token for token in tokens])
    tokens = [token for token in tokens if len(token) > 4]

    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                                                     iterations=500)
        lm_list.append(lm)
        cm = gensim.models.ldamodel.CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())

    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()

    return lm_list, c_v


def reviews_sentiment():

    try:
        df = pd.read_csv("cleanedTextCSV.csv", sep="\t", encoding='latin-1')
    except:
        df = generate_df()

    df = df.dropna()


    stop = stopwords.words('english')
    # df1 = df["cleanedtext"].str.lower().str.split().combine_first(pd.Series([[]], index=df.index))

    for index, row in df.iterrows():
        word_tokens = word_tokenize(row.cleanedtext)

        filtered_sentence = [w for w in word_tokens if not w in stop]

        filtered_sentence = []

        for w in word_tokens:
            if w not in stop:
                filtered_sentence.append(w)

        # print(word_tokens)
        # print(filtered_sentence)

        df.set_value(index, 'cleanedtext', " ".join(filtered_sentence))

    sentiment_scores = list()
    i = 0
    for sentence in df.cleanedtext:
        line = TextBlob(sentence)
        sentiment_scores.append(line.sentiment.polarity)
        # print(sentence + ": POLARITY=" + str(line.sentiment.polarity))

    # df['polarity'] = sentiment_scores
    # normalized_polarity = 2*(df['polarity'] - df['polarity'].min()) / (df['polarity'].max() - df['polarity'].min())-1
    # normalized_score = 2*(df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())-1
    # sns.distplot(normalized_polarity)
    # sns.distplot(normalized_score)
    # plt.show()


    # PLOT POSITIVE
    normalized_polarity = df[df['PosNeg'] == 'positive'].polarity
    normalized_score = (df[df['PosNeg'] == 'positive'].score - df[df['PosNeg'] == 'positive'].score.min()) / (
                df[df['PosNeg'] == 'positive'].score.max() - df[df['PosNeg'] == 'positive'].score.min())+0.5

    sns.distplot(normalized_polarity, kde=False)
    sns.distplot(normalized_score, kde=False)
    #
    normalized_polarity = df[df['PosNeg'] == 'negative'].polarity
    normalized_score = (df[df['PosNeg'] == 'negative'].score - df[df['PosNeg'] == 'negative'].score.min()) / (
            df[df['PosNeg'] == 'negative'].score.max() - df[df['PosNeg'] == 'negative'].score.min())-1

    sns.distplot(normalized_polarity, kde=False)
    sns.distplot(normalized_score, kde=False)

    plt.show()
    #TODO FINE parte negativa e positiva, qua sotto correlazione
    print(np.corrcoef(df.score, df.polarity))
    matplotlib.style.use('ggplot')

    plt.scatter(df.score, df.polarity)
    plt.show()
    # train, test = train_test_split(df, test_size=0.1)
    # train_pos = train[train['sentiment'] == 'positive']
    # train_pos = train_pos['text']
    # train_neg = train[train['sentiment'] == 'negative']
    # train_neg = train_neg['text']


def reviews_absa(productId):
    # https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

    # provo ad importare il df o lo genero
    try:
        df = pd.read_csv("cleanedTextCSV.csv", sep="\t", encoding='latin-1')
    except:
        df = generate_df()

    df = df.dropna()

    # B002QWP89S    629
    # B007M83302    564
    # B0013NUGDE    564
    # B000KV61FC    554
    # B000PDY3P0    486
    # B006N3IG4K    455
    # B003VXFK44    455
    # B001LG945O    347
    # B001LGGH40    338
    # B004ZIER34    330
    # B005K4Q1VI    324
    # B008ZRKZSM    310

    productList = ["B002QWP89S", "B007M83302", "B0013NUGDE", "B000KV61FC", "B000PDY3P0", "B006N3IG4K", "B003VXFK44",
                   "B001LG945O", "B001LGGH40", "B004ZIER34"]

    df = df.loc[df['productid'] == productList[productId]]


    reviews = df.cleanedtext.values
    # grammi
    text_data = []
    for r in reviews:
        tokens = prepare_text_for_lda(r)
        # print(tokens)
        text_data.append(tokens)
    # birammi
    # text_data = []
    # for r in reviews:
    #     tokens = prepare_text_for_lda(r)
    #     print(tokens)
    #     bigram = list(nltk.bigrams(tokens))
    #     tokens = []
    #     for i in bigram:
    #         tokens.append((''.join([w + ' ' for w in i])).strip())
    #     text_data.append(tokens)

    # LDA with Gensim
    # First, we are creating a dictionary from the data,
    # then convert to bag-of-words corpus and save the dictionary and corpus for future use.
    dictionary = corpora.Dictionary(text_data)
    dictionary.filter_extremes(no_below=10, no_above=0.50)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    # Finding out the optimal number of topics
    np.random.seed(50)

    lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=text_data, limit=10)
    max_value = max(c_v)
    max_index = c_v.index(max_value)
    NUM_TOPICS = max_index + 1
    # NUM_TOPICS = 4

    ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary,
                                                       iterations=500)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=6)

    value = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=text_data, dictionary=dictionary,
                                                        coherence='c_v')
    coherence_lda = value.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    x = ldamodel.show_topics(num_topics=NUM_TOPICS, num_words=15, formatted=False)
    topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
    # print(ldamodel.print_topic(2, 100))
    # Below Code Prints Only Words

    sentiment_scores = list()

    # Compute Coherence Score using c_v

    i = 0
    for topic, words in topics_words:
        print(" ".join(words))
        line = TextBlob(" ".join(words))
        sentiment_scores.append(line.sentiment.polarity)
        # print(" ".join(words) + ": POLARITY=" + str(line.sentiment.polarity))

    # for topic in topics:
    #     print(topic)

    index = (NUM_TOPICS * 100) / 2 + 21
    fig = plt.figure(figsize=(20, 10))
    i = 0
    for t in range(NUM_TOPICS):
        i += 1
        # fig.add_subplot(NUM_TOPICS, 1, i)
        ax = plt.subplot(index)
        wordcloud = WordCloud(width=800, height=400).generate(ldamodel.print_topic(t, 10))
        ax.imshow(wordcloud, aspect="equal")
        ax.axis("off")
        index += 1

    plt.suptitle(productList[productId])
    plt.tight_layout(pad=0)
    plt.show()
    k = ldamodel.num_topics
    topicWordProbMat = ldamodel.print_topics(k)
    print(topicWordProbMat)

    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda10 = gensim.models.ldamodel.LdaModel.load('model5.gensim')
    lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=True)

    # plot lda
    #pyLDAvis.show(lda_display10)

    # print("saving LDA...")
    # pyLDAvis.save_html(lda_display10, 'LDA/lda_display10' + productId )
    # print("LDA saved: " + productId)


    print("END ABSA")


if __name__ == "__main__":

    for a in range(9):
        reviews_absa(a)

    reviews_sentiment()


