import numpy as np
import nltk
import collections

# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""



def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # 1: HAM // 0: SPAM
    # return predicted labels of development set

    SPAM = 0
    HAM = 1
    LAPLACE = smoothing_parameter
    PRIOR_HAM = pos_prior
    PRIOR_SPAM = 1 - PRIOR_HAM

    # Counting to find P(Word | Type = Ham) and P(Word | Type = Spam)
    spam_word_counter = collections.Counter()
    ham_word_counter = collections.Counter()
    email_count = np.zeros(2)
    total_token_count = np.zeros(2)
    
    # Count word occurences in each class
    for i in range(len(train_set)):
        email_count[train_labels[i]] = email_count[train_labels[i]] + 1
        for word in train_set[i]:
            total_token_count[train_labels[i]] = total_token_count[train_labels[i]] + 1
            if train_labels[i] == SPAM:
                spam_word_counter.update({word.lower(): 1})
            else:
                ham_word_counter.update({word.lower(): 1})

    # Calculate smoothed likelihoods from counts. The '+1' represents the UNKNOWN probability.
    spam_likelihoods = {key:( (val + LAPLACE)/(total_token_count[SPAM] + 1 + LAPLACE * len(spam_word_counter)) ) for (key, val) in spam_word_counter.items()}
    ham_likelihoods = {key:( (val + LAPLACE)/(total_token_count[HAM] + 1 + LAPLACE * len(ham_word_counter)) ) for (key, val) in ham_word_counter.items()}

    spam_likelihood_sum = sum(spam_likelihoods.values())
    ham_likelihood_sum = sum(ham_likelihoods.values())

    dev_labels = []

    for email in dev_set:
        ham_posterior = np.log(PRIOR_HAM)
        spam_posterior = np.log(PRIOR_SPAM)
        for word in email:
            if word.lower() in spam_likelihoods:
                spam_posterior += np.log(spam_likelihoods[word.lower()])
            else:
                spam_posterior += np.log(1 - spam_likelihood_sum)
            if word.lower() in ham_likelihoods:
                ham_posterior += np.log(ham_likelihoods[word.lower()])
            else:
                ham_posterior += np.log(1 - ham_likelihood_sum)
        
        if ham_posterior >= spam_posterior:
            dev_labels.append(HAM)
        else:
            dev_labels.append(SPAM)

    return dev_labels
    