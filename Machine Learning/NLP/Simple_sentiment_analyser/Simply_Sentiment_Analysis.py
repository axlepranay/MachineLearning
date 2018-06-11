
# Twitter Sentiment Analysis
# We have [Twitter Dataset](https://www.kaggle.com/c/twitter-sentiment-analysis2/data).

# We have 100,000 tweets for training and  300,000 tweets for testing. The Ground truth is 1 for positive tweet and 0 for negative tweet. Let's try to make a sentiment Analyzer using this dataset.

## Load dataset
import pandas as pd
dataFrame = pd.read_csv("train.csv",encoding='latin1')
print(dataFrame)

# Convert data into array
data = dataFrame.values
n = dataFrame.shape[0] ## n is number of tweets
print(n)

##Stored labels and tweets in separate arrays for train data
labels = data[:,1]
tweets = data[:,2]
print(labels.shape)
print(tweets.shape)


# Modify the tweets such that the irrelevant words and characters are removed. To this end apply the following preprocessing.
# 1. **Case** Convert the tweets to lower case.
# 2. **URLs** We don't intend to follow the (short) urls and determine the content of the site, so we can eliminate all of these URLs via regular expression matching or replace it with URL.
# 3. **Username** We can eliminate "$@$username" via regex matching or replace it with AT\_USER
# 4. **hashtag** hash tags can give us some useful information, so replace them with the exact same word without the hash. E.g. \#nike replaced with 'nike'.
# 5. **Whitespace** Replace multiple whitespaces with a single whitespace.
# 6. **Stop words** a, is, the, with etc. The full list of stop words can be found at Stop Word List. These words don't indicate any sentiment and can be removed.
# 7. **Repeated letters** If you look at the tweets, sometimes people repeat letters to stress the emotion. E.g. hunggrryyy, huuuuuuungry for 'hungry'. We can look for 2 or more repetitive letters in words and replace them by 2 of the same.
# 8. **Punctuation** Remove punctuation such as comma, single/double quote, question marks at the start and end of each word. E.g. beautiful!!!!!! replaced with beautiful
# 9. **Non-alpha Words**  Remove all those words which don't start with an alphabet. E.g. 15th, 5.34am

## Preprocess the tweets

## import regex
import re
import numpy as np

#start process_tweet
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    tweet = tweet.strip('.,')
    return tweet


for i in range(n):
    tweets[i] = processTweet(tweets[i])

print(tweets[0:100])

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    for stopWord in open(stopWordListFileName, 'r'):
        stopWords.append(stopWord)
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    words = tweet.split()
    PUNCTUATIONS = '\'"?!,.;:'    
    for w in words:
        # strip punctuation
        w = w.strip(PUNCTUATIONS)
        # check if the word starts with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
 
        #ignore if it is a stop word
        
        if w in stopWords or val is None:
            continue
        else:
            featureVector.append(w.lower())
    return featureVector

def getwordcount(words, count):    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    total = []
    #print words
    
    for w in words:        
        if w in positive_words:
            positive_count += 1
        elif w in negative_words:
            negative_count += 1
        else:
            neutral_count += 1
            
    total.append(positive_count)
    total.append(negative_count)
    total.append(neutral_count)
    total.append(labels[count])
    return total
    
tweets_modified = []
count = 0

stopWords = getStopWordList('stopwords.txt')
positive_words = pd.read_csv('positive-words.txt').values
negative_words=  pd.read_csv('negative-words.txt').values


for i in range(n):
    print(i)
    print(tweets)
    featureVector = getFeatureVector(tweets[i])
    print(featureVector)
    tweets_modified.append(getwordcount(featureVector,count))
   
    count += 1

import numpy as np
x = np.asarray(tweets_modified)
print (x.shape)
print (x[0])

#
### Using features as probabilities
#import matplotlib.pyplot as plt
#plt.figure(1, figsize=(20,10))
#
#colors = ["red","yellow"]
#plt.scatter(x[:,0]/np.sum(x, 1, np.float), x[:,1]/np.sum(x, 1, np.float), c = colors, s=40)
#plt.show()

train_X = x[:70000,:2]
train_Y = x[:70000,3]
test_X = x[70000:,:2]
test_Y = x[70000:,3]
print(train_Y.shape)

## Linear classifier
from sklearn import linear_model
clf = linear_model.SGDClassifier()
clf.fit(train_X,train_Y)
pred_label = (clf.predict(test_X))
print(pred_label)
correct = np.sum(abs(test_Y-pred_label))
print(correct)
accuracy = (correct/np.float(len(test_Y)))*100.0
print(accuracy)
