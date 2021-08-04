import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
# Loading training/validation data and labels from files

train_sents = []
train_labels = []
with open("sentiment-data/train.txt") as f:
    for line in f.readlines():
        line = line.strip()
        train_sents.append(line[1:])
        train_labels.append(int(line[0]))

val_sents = []
val_labels = []
with open("sentiment-data/val.txt") as f:
    for line in f.readlines():
        line = line.strip()
        val_sents.append(line[1:])
        val_labels.append(int(line[0]))

# (a) simple baseline classifier that always predicts label 1

# Creating Predictions arrays of 1 with shape of labels
train_predictions = np.ones(len(train_labels))
val_predictions = np.ones(len(val_labels))

# Calculating Accuracies
training_accuracy = np.mean(np.where(train_labels == train_predictions, 1, 0))
val_accuracy = np.mean(np.where(val_labels == val_predictions, 1, 0))

print("\nSimple baseline model that always predicts labels as 1\n")
print("Training accuracy ", round(training_accuracy*100, 2), '%')
print("Validation accuracy ", round(val_accuracy*100, 2), '%\n')

# (b) Implementing a Naive Bayes Classifier

# Learning Vocabulary and Vectorizing training data
vect = CountVectorizer()
vect.fit(train_sents)
train_data = vect.fit_transform(train_sents)

# Creating Multinomial naive Bayes Classifier from Scikit Learn Library
clf = MultinomialNB()

# Training using training data and labels and obtaining accuracy
clf.fit(train_data.toarray(), train_labels)

print("Multinomial Naive Bayes model.\n")

print("Training Accracy", round(
    clf.score(train_data.toarray(), train_labels)*100, 2), '%')

# Vectorizing Validation data and testing it
val_data = vect.transform(val_sents)

print("Validation Accuracy", round(
    clf.score(val_data.toarray(), val_labels)*100, 2), '%')

# (c) Implementing Logistic Regression Model


clf = LogisticRegression(max_iter=200)

clf.fit(train_data.toarray(), train_labels)

print("\nLogistic Regression model.\n")

print("Training Accuracy ", round(
    clf.score(train_data.toarray(), train_labels)*100, 2), '%')

print("Validation Accuracy ", round(
    clf.score(val_data.toarray(), val_labels)*100, 2), '%')

# Training accuracy is higher then validation compared to Multinomial Naive Bayes. Kind of overfitting.

# (d) Implementing Logistic Regression Model with Bigram Features

vect2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
vect2.fit(train_sents)
train_bigram_data = vect2.fit_transform(train_sents)

clf = LogisticRegression(solver='liblinear')

clf.fit(train_bigram_data.toarray(), train_labels)

print("\nLogistic Regression model with bigrams.\n")

print("Training Accuracy ", round(
    clf.score(train_bigram_data.toarray(), train_labels)*100, 2), '%')

val_bigram_data = vect2.transform(val_sents)

print("Validation Accuracy ", round(
    clf.score(val_bigram_data.toarray(), val_labels)*100, 2), '%')

# Training is more higher then validation compared to unigram logistic regression. More Overfitting
