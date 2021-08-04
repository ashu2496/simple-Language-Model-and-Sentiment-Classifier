# Dependencies nltk, pandas, matplotlib, collections

import math
import nltk
import string
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize

# (a) Loading data from file into variable and tokenizing it
fileobject = open(r"lm-data/brown-train.txt", "r")
train_text = fileobject.read()
fileobject.close()
fileobject = open(r"lm-data/brown-val.txt", "r")
val_text = fileobject.read()
fileobject.close()

# Removing punctuation and tokenizing data
train_text = word_tokenize(train_text)
val_text = word_tokenize(val_text)

# (b) Counting and plotting frequencies of vocabulary from training data

# Counting and sorting accourding to count

frequency = Counter(train_text).most_common()

# Converting into dataframe for plotting
graph = pd.DataFrame(frequency, columns=['words', 'count'])
print(graph)

# Plotting dataframe into line graph
fig, ax = plt.subplots(figsize=(12, 8))
graph.plot(kind='line', title="Frequency of each word",
           ax=ax, x='words', y='count')
plt.show()
# Top 50 to 100 have high frequencies and later have very few, it does follow zipf's law

# (c) Computing bigram probabilities and calculate perplexities

# Count_probabilities will get probabilities from bigrams and unigrams


def count_probabilities(unigrams, alpha):

    # Dictionary  Placeholder for calculating probability
    bigram_probability = {}

    # Calculating frequencies of bigrams and unigrams
    unigram_frequency = dict(Counter(unigrams).most_common())
    bigrams = list(nltk.bigrams(unigrams))
    bigram_frequency = dict(Counter(bigrams).most_common())

    # Calculating probability of each bigram and saving it into placeholder created above
    for bigram in bigram_frequency:

        # probability = bigram_frequency + alpha / unigram_frequency + (alpha * len(unigrams))
        bigram_probability[bigram] = (
            bigram_frequency[bigram] + alpha) / (unigram_frequency[bigram[0]] + (alpha * len(unigram_frequency)))

    return bigram_probability


''' calculate_perplexity will calculate perplexities from
    probabilities (will use traing data's probabilities)'''


def calculate_perplexity(unigrams, bigram_probability, number_of_words):

    bigrams = list(nltk.bigrams(unigrams))

    # lg = Sum (Log2(p(Wi/Wi-1)))
    lg = 0
    for element in bigrams:
        lg += math.log2(bigram_probability[element])

    # 2^(-1/N)* lg
    ppl = 2**((-1/number_of_words) * lg)

    return ppl

# Taking Probabilities of validation set from training set


def val_probabilities(unigrams, training_probability, val_unigrams, alpha):
    val_probability = {}

    # Calculating training Unigrams frequency
    unigrams_frequency = dict(Counter(unigrams).most_common())

    # Obtaining Unique Validation Bigrams
    val_bigrams = list(set(list(nltk.bigrams(val_unigrams))))

    # Getting Probaillity of each unique bigram
    for bigram in val_bigrams:

        ''' Obtaing Validation bigrams probabillity from training bigram's probabillities. if not found then
        Validation bigram probabillity = (0 + alpha) / 
        (unigram count from traning unigram frequency (0 if not found) + (alpha * Number of unique words in training data)) '''

        val_probability[bigram] = training_probability.get(
            bigram, (alpha)/(unigrams_frequency.get(bigram[0], 0) + (alpha*len(unigrams_frequency))))

    return val_probability

# Calculating training and validation Perplexity for each alpha and plotting in terms of alpha


list_alpha = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
train = {}
val = {}

for item in list_alpha:

    # Calculating training bigram probabilities
    bigram_probability = count_probabilities(train_text, alpha=item)

    # calculating training perplexity from probabilities obtained above
    perplexity = calculate_perplexity(
        train_text, bigram_probability, len(train_text))

    print("Training Perplexity using alpha = ",
          item, " is ", perplexity)

    # Saving perplexities for graph
    train[item] = perplexity

    # Obtaining Validation probailities from training probaiblities using function val_probabilities
    val_probability = val_probabilities(
        train_text, bigram_probability, val_text, alpha=item)

    # Calculating validation perplexity from obtained probabilities above
    perplexity = calculate_perplexity(
        val_text, val_probability, len(val_text))

    print("Validation Perplexity using alpha = ",
          item, " is ", perplexity)

    # Saving perplexities for graph
    val[item] = perplexity

# Plotting Graph
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Training and Validation Perplexities')
plt.xlabel('Alpha')
plt.ylabel('Perplexity')
plt.plot(list(train.keys()),
         list(train.values()), label='training perplexity', marker='o')
plt.plot(list(val.keys()),
         list(val.values()), label='validation perplexity', marker='*')
plt.legend()
plt.show()

# Training perplexity is lower and rises with alpha wheareas validation is highr first and it gets lower and rises again.

''' (d) fix alpha = 0.001 and vary training data used from 10% to 100% in increament of 10% and plot training
 and validation perplexities in terms of fraction of data'''

alpha = 0.001

training_graph_perplexity = {}
val_graph_perplexity = {}

for i in range(1, 11):

    # Creating fractions of training data
    fraction = round(i*1e-1, 1)
    fraction_train_text = train_text[:int(fraction*len(train_text))]

    # Calculating bigram probabilities of fractioned training data
    bigram_probability = count_probabilities(fraction_train_text, alpha=alpha)

    # calculating training perplexity from probabilities obtained above
    perplexity = calculate_perplexity(
        fraction_train_text, bigram_probability, len(fraction_train_text))

    print("Training Perplexity using alpha = ",
          alpha, " and ", i*10, '%', "of training data is ", perplexity)

    # Savign Perplexity for graph
    training_graph_perplexity[i*0.1] = perplexity

    # Obtaining Validation probailities from training probaiblities using function val_probabilities
    val_probability = val_probabilities(
        fraction_train_text, bigram_probability, val_text, alpha=alpha)

    # Calculating validation perplexity from obtained probabilities above
    perplexity = calculate_perplexity(
        val_text, val_probability, len(val_text))

    print("Validation Perplexity using alpha = ",
          alpha, " and ", i*10, '%', "of training data is ", perplexity)

    # Saving Perplexity for graph
    val_graph_perplexity[i*0.1] = perplexity

# Plotting Graph

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Training and Validation Perplexities')
plt.xlabel('Fracton of training data')
plt.ylabel('Perplexity')
plt.plot(list(training_graph_perplexity.keys()),
         list(training_graph_perplexity.values()), label='training perplexity', marker='o')
plt.plot(list(val_graph_perplexity.keys()),
         list(val_graph_perplexity.values()), label='validation perplexity', marker='*')
plt.legend()
plt.show()

# With less fraction of training data the validation perplexity is higher and its gets lower wit hhigh fraction of training data.
