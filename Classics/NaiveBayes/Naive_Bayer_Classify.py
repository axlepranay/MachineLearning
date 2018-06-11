 
# ## Naive Bayes ##


import pandas as pd
import random
import numpy as np


# Read the dataset
data = pd.read_csv("pima-indians-diabetes.csv",header = None)

# Visualize some data samples from the dataset
print('Pima Indians Diabetes Data Set\n')
print(data.head())


# 8th column is the class label
print('\n\n\nStats for the 7 features over the dataset and the 2 classes {8th column}{diabetic/not-diabetic}\n')
print(data.describe())


TRAIN_TEST_RATIO = 0.8        # 80% training data
picker = list(range(data.shape[0]))        # get all indices as a list
## sometimes the data is arranged classwise and not randomly
## therefore we shuffle the indices
random.shuffle(picker)
trainMax = int(data.shape[0] * TRAIN_TEST_RATIO)

train_features = []
test_features = []
train_labels = []
test_labels = []

for pick in picker[:trainMax]:
    train_features.append(data.values[pick][:-1])
    train_labels.append(int(data.values[pick][-1]))
for pick in picker[trainMax:]:
    test_features.append(data.values[pick][:-1])
    test_labels.append(int(data.values[pick][-1]))

train_features = np.array(train_features)
test_features = np.array(test_features)

data.values[pick]

print(train_features.shape, len(train_labels), test_features.shape, len(test_labels))


# ### Classifying Diabetes
# 
# We have features and labels present in the dataset. Now we need to learn naive bayes classifier for this dataset.
# 
# $$ P(y|X) = \frac{P(X|y)*P(y)}{P(X)} \propto (\Pi_{i}P(X_{i}|y))*P(y) $$
# 
# Here, 'y' represents a class, $ X_{i} $ represents the $ i^{th} $ feature and X is the set of all features.

# Get the number of unique classes & corresponding number of elements belonging to each class
classes, counts = np.unique(train_labels, return_counts=True)
print(classes)
print(counts)


### I assume my classes are from 0 ... N for some N (Here, we have just 2 classes)
num_classes = len(classes)
num_feats = train_features.shape[1]  #total number of features
total_samples = len(train_labels)    #total number of samples

# Prior for any class = {number of samples belonging to that class/ total_samples}
prior = np.array([ x*1.0/total_samples for x in counts ])

print(prior)


# ### Finding Posterior Distribution $P(X| y)$ or $\Pi_{i}P(X_{i}| y)$ from the data
# 
# This is a little complicated. For each feature $X_{i}$, we assume that they are uncorrelated (Naive Bayes Assumption) and thus we can write,
# 
# $$ P(X| y) = \Pi_{i} P(X_{i} | y) $$
# 
# We wish to estimate $P(X_{i} | y)$ for each $X_{i}$. Thus for samples corresponding to each class label $y$, we need to estimate the probability distribution. Here, however, our features are real values. So we will assume the distribution to be gaussian and estimate the parameters corresponding to each feature using all training samples per class. Thus, we need to calculate mean (say $ m_{y, i} $) and standard deviation (say $ v_{y, i} $) for each class $y$ and each feature $i$.

### Calculate the mean and variance per feature dimension here 
### from the training set from samples belonging to each class label.

###  Mean is the just the sum of a vector divided by the length of vector
### Variance is the square of standard deviation.

means = np.zeros((num_feats, num_classes)) # every feature, for each class
stddev = np.zeros((num_feats, num_classes)) # every feature, for each class

# For each class
for y in classes: # selecting a class 'y'
    pts = train_features[np.where( train_labels == y )[0], :]    # get all samples belonging to 'y'
    # For each feature
    for i in range(num_feats):
        means[i, y] = np.mean(pts[:, i])
        stddev[i, y] = np.std(pts[:, i])

### This completes the training phase
### We know have estimated both the prior probability and the posterior distributions from our training set.


# ### Combine prior and posterior to classify points.
# 
# Now, given the mean 'm' and standard deviation 'v', the posterier probability is estimated by the normal distribution (also called gaussian distribution) using:
# 
# $$ P(X_{i}| y) = \frac{1}{\sqrt{2\pi v_{y, i}^{2}}} exp( - \frac{ (X_{i} - m_{y, i})^{2} }{2v_{y, i}^{2}} ) $$ 
# 
# 
# Use the formula given above and the parameters computed earlier, we predict the likelihood of every sample in a test set to belong to each class. $\Pi$ stands for multiplication of each $P(X_{i}|y)$ ($i$ is the $i^{th}$ feature).
# 
# Note: One may use other kinds of distributions too.

def gaussian(x, m, v):
    g = np.sqrt(1.0/2*np.pi*v*v)*np.exp( -1.0*(((x - m)/v)**2) )
    return g

# Get likelihood
def get_likelihood(point, means, stddev):
    
    feat_prob = np.zeros((num_feats, num_classes))
    for y in classes:
        for i in range(num_feats):
            feat_prob[i, y] = gaussian(point[i], means[i, y], stddev[i, y]) # get the probability
    
    likelihood = np.zeros((num_classes, 1)) # likelihood for each class 'y'
    for y in classes:
        # Take the product of all the feature likelihoods of the class considered
        likelihood[y] = np.prod(feat_prob[np.nonzero(feat_prob), y]) # mutliply for each feature 'Xi'
    
    return likelihood


## Predict using Naive Bayes classifier


predictions = []
# For each test sample
for i in range(len(test_labels)):
    
    # Get its likelihood of belong to either class
    likelihood = get_likelihood(test_features[i, :], means, stddev)
    
    # Calculate the approximate posterior = likelihood * prior
    approx_posterior = [ np.asscalar(x*y) for x,y in zip(likelihood, prior) ]
    #approx because of missing P(X) (constant) in the denominator
    
    # Make the prediction as that class with the maximum approximate posterior
    prediction = np.argmax(approx_posterior)
    predictions.append(prediction)

print("Accuracy")
print(np.mean([x == y for x, y in zip(predictions, test_labels)]))

