import pandas as pd
import numpy as np
from collections import namedtuple, Counter
# import tensorflow as tf
from string import punctuation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
# from sklearn.metrics import roc_curve, auc, classification_report




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


    # creo un nuovo attributo Sentiment
    # negative = review(1-3), positive=(4-5)
    df.ix[df.score > 3, 'Sentiment'] = "POSITIVE"
    df.ix[df.score <= 3, 'Sentiment'] = "NEGATIVE"


    ## Sentiment distribution
    # ax = plt.axes()
    # sns.countplot(df.Sentiment, ax=ax)
    # ax.set_title('Sentiment Positive vs Negative Distribution')
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

    print("Proportion of positive review:", len(df[df.Sentiment == "POSITIVE"]) / len(df))
    print("Proportion of positive review:", len(df[df.Sentiment == "NEGATIVE"]) / len(df))
    # 77% of the fine food reviews are considered as positive
    # and 23% of them are considered as negative.

    reviews = df.text.values
    labels = df.Sentiment.values

    ### TEXT REVIEWS ###

    if df.Sentiment[1] == "POSITIVE":
        print("POSITIVE" + "\t" + reviews[1][:90] + "...")
    else:
        print("NEGATIVE" + "\t " + reviews[1][:90] + "...")


    ### Exploratory Visualization ###
    # mi creo un vocabolario di parole positive e negative

    positive_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "POSITIVE"]
    negative_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "NEGATIVE"]

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




    #### POSITIVE WORDCLOUD #####
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



    #### NEGATIVE WORDCLOUD #####
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





def data_preprocessing(df):

    reviews = df.text.values
    labels = np.array([1 if s == "POSITIVE" else 0 for s in df.Sentiment.values])

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









if __name__ == "__main__":
    df = pd.read_csv("Dataset/food.tsv", sep="\t", encoding='latin-1')
    exploratory_data_analysis(df)
    data_preprocessing(df)
