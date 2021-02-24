# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import collections

def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda, unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # HAM: 1 // SPAM: 0
    # return predicted labels of development set

    SPAM = 0
    HAM = 1
    LAMBDA = bigram_lambda
    LAPLACE_UNI = unigram_smoothing_parameter
    LAPLACE_BI = bigram_smoothing_parameter
    PRIOR_HAM = pos_prior
    PRIOR_SPAM = 1 - PRIOR_HAM

    # START UNIGRAM TRAIN
    uni_spam_word_counter = collections.Counter()
    uni_ham_word_counter = collections.Counter()
    uni_email_count = np.zeros(2)
    uni_total_token_count = np.zeros(2)

    # Count word occurences in each class
    for i in range(len(train_set)):
        uni_email_count[train_labels[i]] = uni_email_count[train_labels[i]] + 1
        for word in train_set[i]:
            uni_total_token_count[train_labels[i]] = uni_total_token_count[train_labels[i]] + 1
            if train_labels[i] == SPAM:
                uni_spam_word_counter.update({word: 1})
            else:
                uni_ham_word_counter.update({word: 1})

    # Calculate smoothed likelihoods from counts. The '+1' represents the UNKNOWN probability.
    uni_spam_likelihoods = {key:( (val + LAPLACE_UNI)/(uni_total_token_count[SPAM] + 1 + LAPLACE_UNI * len(uni_spam_word_counter)) ) for (key, val) in uni_spam_word_counter.items()}
    uni_ham_likelihoods = {key:( (val + LAPLACE_UNI)/(uni_total_token_count[HAM] + 1 + LAPLACE_UNI * len(uni_ham_word_counter)) ) for (key, val) in uni_ham_word_counter.items()}

    uni_spam_likelihood_sum = sum(uni_spam_likelihoods.values())
    uni_ham_likelihood_sum = sum(uni_ham_likelihoods.values())
    # END UNIGRAM TRAIN

    # START BIGRAM TRAIN
    bi_spam_word_counter = collections.Counter()
    bi_ham_word_counter = collections.Counter()
    bi_email_count = np.zeros(2)
    bi_total_token_count = np.zeros(2)
    
    # Count bigram occurences in each class
    for i in range(len(train_set)):
        bi_email_count[train_labels[i]] = bi_email_count[train_labels[i]] + 1
        for j in range(len(train_set[i]) - 1):
            bigram = (train_set[i][j], train_set[i][j + 1])
            bi_total_token_count[train_labels[i]] = bi_total_token_count[train_labels[i]] + 1
            if train_labels[i] == SPAM:
                bi_spam_word_counter.update({bigram: 1})
            else:
                bi_ham_word_counter.update({bigram: 1})

    # Calculate smoothed likelihoods from counts. The '+1' represents the UNKNOWN probability.
    bi_spam_likelihoods = {key:( (val + LAPLACE_BI)/(bi_total_token_count[SPAM] + 1 + LAPLACE_BI * len(bi_spam_word_counter)) ) for (key, val) in bi_spam_word_counter.items()}
    bi_ham_likelihoods = {key:( (val + LAPLACE_BI)/(bi_total_token_count[HAM] + 1 + LAPLACE_BI * len(bi_ham_word_counter)) ) for (key, val) in bi_ham_word_counter.items()}

    bi_spam_likelihood_sum = sum(bi_spam_likelihoods.values())
    bi_ham_likelihood_sum = sum(bi_ham_likelihoods.values())
    # END BIGRAM TRAIN

    dev_labels = []

    for email in dev_set:
        ham_posterior = np.log(PRIOR_HAM)
        spam_posterior = np.log(PRIOR_SPAM)

        # Unigram 
        uni_ham_sum = 0
        uni_spam_sum = 0
        for word in email:
            if word in uni_spam_likelihoods:
                uni_spam_sum += np.log(uni_spam_likelihoods[word])
            else:
                uni_spam_sum += np.log(1 - uni_spam_likelihood_sum)
            if word in uni_ham_likelihoods:
                uni_ham_sum += np.log(uni_ham_likelihoods[word])
            else:
                uni_ham_sum += np.log(1 - uni_ham_likelihood_sum)

        # Bigram
        bi_ham_sum = 0
        bi_spam_sum = 0
        for i in range(len(email) - 1):
            bigram = (email[i], email[i + 1])
            if bigram in bi_spam_likelihoods:
                bi_spam_sum += np.log(bi_spam_likelihoods[bigram])
            else:
                bi_spam_sum += np.log(1 - bi_spam_likelihood_sum)
            if bigram in bi_ham_likelihoods:
                bi_ham_sum += np.log(bi_ham_likelihoods[bigram])
            else:
                bi_ham_sum += np.log(1 - bi_ham_likelihood_sum)
        
        ham_posterior += (1-LAMBDA) * uni_ham_sum + LAMBDA * bi_ham_sum
        spam_posterior += (1-LAMBDA) * uni_spam_sum + LAMBDA * bi_spam_sum

        if ham_posterior >= spam_posterior:
            dev_labels.append(HAM)
        else:
            dev_labels.append(SPAM)

    return dev_labels