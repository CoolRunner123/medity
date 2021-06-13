# Import PyDrive and associated libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import emoji
#import contractions
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from expertai.nlapi.cloud.client import ExpertAiClient



df = pd.read_csv("tweet_data.csv")
print(df.sample(10))
print("Number of tweets: {}".format(len(df)))

tweet_id = 4879
tweet = df.iloc[tweet_id]
print("Tweet: {}".format(tweet["tweet_text"]))
print("Tweet sentiment: {}".format(tweet["sentiment"]))


sentiment_count = df["sentiment"].value_counts()
plt.pie(sentiment_count, labels=sentiment_count.index,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()


print("Number of + tweets: {}".format(df[df["sentiment"]=="positive"].count()[0]))
print("Number of - tweets: {}".format(df[df["sentiment"]=="negative"].count()[0]))

tweet = "I love this! "

#replace the RT with ""
def replace_retweet(tweet, default_replace=""):
  tweet = re.sub('RT\s+', default_replace, tweet)
  return tweet

#replace the person who tweeted name
def replace_user(tweet, default_replace="twitteruser"):
  tweet = re.sub('\B@\w+', default_replace, tweet)
  return tweet

print("Processed tweet: {}".format(replace_retweet(tweet)))

#turn the emoji into text
def demojize(tweet):
  tweet = emoji.demojize(tweet)
  return tweet

#replace the https
def replace_url(tweet, default_replace=""):
  tweet = re.sub('(http|https):\/\/\S+', default_replace, tweet)
  return tweet

#replace the hashtags
def replace_hashtag(tweet, default_replace=""):
  tweet = re.sub('#+', default_replace, tweet)
  return tweet

tweet = "LOOOOOOOOK at this ... I'd like it so much!"

#lowercase everything
def to_lowercase(tweet):
  tweet = tweet.lower()
  return tweet

#get rid of word repetition
def word_repetition(tweet):
  tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
  return tweet

#replace multiple punctuation with single 
def punct_repetition(tweet, default_replace=""):
  tweet = re.sub(r'[\?\.\!]+(?=[\?\.\!])', default_replace, tweet)
  return tweet

# #fix contractions
# def _fix_contractions(tweet):
#   for k, v in contractions.contractions_dict.items():
#     tweet = tweet.replace(k, v)
#   return tweet

# #fix contractions using package
# def fix_contractions(tweet):
#   tweet = contractions.fix(tweet)
#   return tweet

nltk.download('punkt')

tweet = "These are 5 different words!"
#return list of tokens
def tokenize(tweet):
  tokens = word_tokenize(tweet)
  return tokens

print(type(tokenize(tweet)))
print("Tweet tokens: {}".format(tokenize(tweet)))

#irrelevant words
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
#keep the word not
stop_words.discard('not')

#custom token function
def custom_tokenize(tweet,
                    keep_punct = False,
                    keep_alnum = False,
                    keep_stop = False):
  
  token_list = word_tokenize(tweet)

  if not keep_punct:
    token_list = [token for token in token_list
                  if token not in string.punctuation]

  if not keep_alnum:
    token_list = [token for token in token_list if token.isalpha()]
  
  if not keep_stop:
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')
    token_list = [token for token in token_list if not token in stop_words]

  return token_list


  #list of tokens to stem
  tokens = ["manager", "management", "managing"]

  #define stemmers
  porter_stemmer = PorterStemmer()
  lancaster_stemmer = LancasterStemmer()
  snoball_stemmer = SnowballStemmer('english')

def stem_tokens(tokens, stemmer):
    token_list = []
    for token in tokens:
        token_list.append(stemmer.stem(token))
    return token_list

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens, word_type, lemmatizer):
  token_list = []
  for token in tokens:
    token_list.append(lemmatizer.lemmatize(token, word_type[token]))
  return token_list


#putting it all together
def process_tweet(tweet, verbose=False):
  if verbose: print("Initial tweet: {}".format(tweet))

  ## Twitter Features
  tweet = replace_retweet(tweet) # replace retweet
  tweet = replace_user(tweet, "") # replace user tag
  tweet = replace_url(tweet) # replace url
  tweet = replace_hashtag(tweet) # replace hashtag
  if verbose: print("Post Twitter processing tweet: {}".format(tweet))

  ## Word Features
  tweet = to_lowercase(tweet) # lower case
  tweet = punct_repetition(tweet) # replace punctuation repetition
  tweet = word_repetition(tweet) # replace word repetition
  tweet = demojize(tweet) # replace emojis
  if verbose: print("Post Word processing tweet: {}".format(tweet))

  ## Tokenization & Stemming
  tokens = custom_tokenize(tweet, keep_alnum=False, keep_stop=False) # tokenize
  stemmer = SnowballStemmer("english") # define stemmer
  stem = stem_tokens(tokens, stemmer) # stem tokens

  return stem


df["tokens"] = df["tweet_text"].apply(process_tweet)
df["tweet_sentiment"] = df["sentiment"].apply(lambda i: 1
                                              if i == "positive" else 0)

X = df["tokens"].tolist()
y = df["tweet_sentiment"].tolist()
# each group of tokens will be categorized as a 0 (neg) or 1 (pos)
#print(X)
#print(y)
print("hiiiiii")

corpus = [["i", "love", "nlp"],
          ["i", "miss", "you"],
          ["i", "love", "you"],
          ["you", "are", "happy", "to", "learn"],
          ["i", "lost", "my", "computer"],
          ["i", "am", "so", "sad"]]

sentiment = [1, 0, 1, 1, 0, 0]

def build_freqs(tweet_list, sentiment_list):
  freqs = {}
  for tweet, sentiment in zip(tweet_list, sentiment_list):
    for word in tweet:
      pair = (word, sentiment)
      if pair in freqs:
        freqs[pair] += 1
      else:
        freqs[pair] = 1
  return freqs

#freqs = build_freqs(corpus, sentiment)
#freqs_all = build_freqs(X, y)

freqs_all = build_freqs(X, y)



def tweet_to_freq(tweet, freqs):
  x = np.zeros((2,))
  for word in tweet:
    if (word, 1) in freqs:
      x[0] += freqs[(word, 1)]
    if (word, 0) in freqs:
      x[1] += freqs[(word, 0)]
  return x

  print(tweet_to_freq(["i", "love", "nlp"], freqs))


def fit_tfidf(tweet_corpus):
  tf_vect = TfidfVectorizer(preprocessor=lambda x: x,
                            tokenizer=lambda x: x)
  tf_vect.fit(tweet_corpus)
  return tf_vect

X_train, X_test, y_train,y_test = train_test_split(X,y,random_state=0,
                                                   train_size = 0.80)
tf = fit_tfidf(X_train)
X_train_tf = tf.transform(X_train)
X_test_tf = tf.transform(X_test)

def fit_lr(X_train,y_train):
  model = LogisticRegression()
  model.fit(X_train,y_train)
  return model

model_lr_tf = fit_lr(X_train_tf,y_train)

your_tweet = "I felt so stressed today. I am preparing for my interview tomorrow so I am nervous."

def predict_tweet(tweet):
    processed_tweet = process_tweet(tweet)
    print(processed_tweet)
    transformed_tweet = tf.transform([processed_tweet])
    print(transformed_tweet)
    prediction = model_lr_tf.predict(transformed_tweet)

    print(prediction)

    if prediction == 1:
        return "Prediction is positive sentiment"
    else:
         df = pd.read_csv("quotes.csv")
         print(df.sample(1))
         return "Prediction is negative sentiment"
        
         

print(predict_tweet(your_tweet))

negativewords = 0
possitivewords= 0
totalwords = 0
for word in process_tweet(your_tweet):
    totalwords = totalwords +1
    if freqs_all[(word, 1)] > freqs_all[(word, 0)]:
        possitivewords = possitivewords +1
    else:
        negativewords = negativewords + 1


print(negativewords/totalwords)
print("Ratio")
print(negativewords/possitivewords)

print("Frequency of word 'love' in + tweets: {}".format(freqs_all[("hate", 1)]))
print("Frequency of word 'love' in - tweets: {}".format(freqs_all[("hate", 0)]))



client = ExpertAiClient()

text = "I felt so stressed today. I am preparing for my interview tomorrow so I am nervous." 
language= 'en'

output = client.specific_resource_analysis(
    body={"document": {"text": text}}, 
    params={'language': language, 'resource': 'sentiment'
})

# Output overall sentiment


print("Output overall sentiment:")

print(output.sentiment.overall)