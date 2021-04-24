#sentiment analysis
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import pandas as pd
import re, string, random
import pickle

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


if __name__ == "__main__":
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
    stop_words = stopwords.words('english')
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = [] #empty cleaned tokens list
    negative_cleaned_tokens_list = [] #empty cleaned tokens list

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive") #put all positive tweets in one set
                     for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative") #put all negative tweets in one set
                     for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset #combine twitter dataset so that we have the original whole


    random.shuffle(dataset) #shuffle to remove bias

    train_data = dataset[:7000]
    test_data = dataset[7000:]
    classifier_f = open("naivebayes.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()

    print("Accuracy is:", classify.accuracy(classifier, test_data), 'for training data')

    test_dataset_raw = pd.read_csv('G:\Anaconda\sms-20210328134858.xml\sms3.csv', usecols=['Conversation', 'DateTime', 'MessageType', 'Body'])
    test_dataset_clean = test_dataset_raw[['DateTime', 'MessageType', 'Body']] #remove Conversation column from dataset
    test_dataset_clean.dropna(axis=0, inplace=True)

    # message = ['The quick brown fox jumps over the fence', 'Yeet the fetus']
    test_dataset_clean_set = set([])
    passage_dict = dict()
    for text in test_dataset_clean['Body'].to_list():
        test_dict = dict()
        for word in word_tokenize(text):
            # print(word.lower())
            if word.lower() in test_dict:
                continue
            else:
                test_dict[word.lower()] = True
            # for i in test_dataset_clean_set:
                #print(classifier.classify(test_dict)
        # if passage_dict[text]:
        #     continue
        # else:
        passage_dict[text] = classifier.classify(test_dict)
    # print(passage_dict)

# classified = pd.DataFrame.from_records(data=passage_dict, index=[0])
classified = pd.DataFrame.from_dict(data=passage_dict, orient = 'index')
classified.columns = ['output']

merged_df = classified.merge(test_dataset_clean, left_on=classified.index, right_on='Body')

print(merged_df.head())

# to find positive texts only
#merged_df[merged_df['Output'] == 'Positive']

# Positive vs negative text (who sent more)

sumPositive = merged_df.groupby('output').count()
sums_Positive = sumPositive.groupby[sumPositive[sumPositive.index == 'Positive'],sumPositive['MessageType']]

print(sums_Positive)
#clean_dates = pd.to_datetime(sum_Positive['DateTime'])
#month_totals = clean_dates.groupby(['DateTime'].month)['output'].count()
#
# print(month_totals)
# Was the proportion of positive texts higher
# or lower at different times of the day/month/year?



    #my_test_data = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in test_dataset_clean]




    #test_dict = dict([token, True] for token in test_dataset_clean_set)
    #print(test_dict)
    #x = classifier.classify(dict([token, True] for token in test_dataset_clean_set))
    #print(x)
    #print(classifier.show_most_informative_features(10))
    #print(classifier.show_most_informative_features(10))

