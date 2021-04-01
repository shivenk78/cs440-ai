# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

from collections import Counter
import numpy as np
import math

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    LAPLACE = 0.001
    UNKNOWN_WORD = '__UNKNOWN__'

    tag_count = Counter()
    first_tag_count = Counter()
    tag_transition_count = Counter()
    emission_count_dict = {}
    unique_words = set()

    total_word_count = 0

    for sentence in train:
        # Pairwise counts
        for i in range(len(sentence)-1):
            word1, tag1 = sentence[i]
            word2, tag2 = sentence[i+1]

            if i == 0:
                first_tag_count.update({tag1 : 1})
            tag_transition_count.update({(tag1, tag2) : 1})
        
        # Individual word counts
        for word, tag in sentence:
            if tag not in emission_count_dict:
                emission_count_dict[tag] = Counter()
            emission_count_dict[tag].update({word : 1})

            tag_count.update({tag: 1})

            unique_words.add(word)

            total_word_count += 1

    unique_words.add(UNKNOWN_WORD)
    NUM_UNIQUE_WORDS = len(unique_words)
    TAGS = [tag for tag in tag_count]

    # tag : prob
    initial_probs = {tag: math.log((first_tag_count[tag] + LAPLACE)/ (len(train) + 1 + LAPLACE*len(tag_count))) for tag in tag_count}

    # (tag1, tag2) : prob # Laplace smoothing, including unseen transition combos
    transition_probs = {(tag1, tag2): math.log((tag_transition_count[(tag1, tag2)] + LAPLACE) / (tag_count[tag1] + LAPLACE*(1+len(tag_count)))) for tag1 in tag_count for tag2 in tag_count}

    # tag : {word : prob}
    emission_probs = {} 
    for tag in TAGS:
        emission_probs[tag] = {}
        for word in unique_words:
            emission_probs[tag][word] = math.log( (emission_count_dict[tag][word] + LAPLACE) / (tag_count[tag] + LAPLACE * NUM_UNIQUE_WORDS) )

    #print(emission_probs)

    # BEGIN CALCULATE
    output = []
    #ITER = 0

    for sentence in test:
        Vit = np.zeros((len(TAGS), len(sentence)))
        backpointer_row = np.zeros(Vit.shape, dtype=int)
        
        # Trellis initial values (first column)
        for i in range(len(TAGS)):
            if sentence[0] in emission_probs[TAGS[i]]:
                Vit[i][0] = initial_probs[TAGS[i]] + emission_probs[TAGS[i]][sentence[0]]
            else:
                Vit[i][0] = initial_probs[TAGS[i]] + emission_probs[TAGS[i]][UNKNOWN_WORD]
        
        # VITERBI PROPAGATION - col is word, row is tag
        for col in range(1, Vit.shape[1]): # words / time axis
            curr_word = sentence[col]
            for row in range(Vit.shape[0]): # each possible tag (states)
                poss_vals = np.zeros(Vit.shape[0])
                this_emiss = emission_probs[TAGS[row]][curr_word] if curr_word in emission_probs[TAGS[row]] else emission_probs[TAGS[row]][UNKNOWN_WORD]
                for prev_row in range(Vit.shape[0]): # each possible previous tag
                    # prev node + transition + emission
                    poss_vals[prev_row] = Vit[prev_row][col-1] + transition_probs[TAGS[prev_row], TAGS[row]] + this_emiss

                best_row = np.argmax(poss_vals)
                Vit[row][col] = poss_vals[best_row]
                #print(poss_vals)
                backpointer_row[row, col] = best_row

        # BACKTRACK
        overall_best_row = np.argmax(Vit[:, -1])
        #print(overall_best_row)
        row = overall_best_row
        this_output = []
        for col in reversed(range(1, Vit.shape[1])):
            this_output.append((sentence[col], TAGS[row]))
            row = backpointer_row[row, col]
        this_output.append((sentence[0], TAGS[row]))
        this_output.reverse()
        # print(Vit)
        # print(this_output)
        # print(backpointer)
        output.append(this_output)

        # ITER += 1
        # if ITER % 100==0:
        #     print(ITER/len(test))

    return output