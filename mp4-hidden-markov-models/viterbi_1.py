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

    tag_count = Counter()
    first_tag_count = Counter()
    tag_transition_count = Counter()
    emission_count_dict = {}

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

            total_word_count += 1

    # tag : prob # <only contains 'START':1.0>
    initial_probs = {tag:first_tag_count[tag] / len(train) for tag in first_tag_count}

    # (tag1, tag2) : prob
    transition_probs = {(tags[0],tags[1]):tag_transition_count[tags] / tag_count[tags[0]] for tags in tag_transition_count}

    # tag : {word : prob}
    emission_probs = {} 
    for tag, words in emission_count_dict.items():
        if tag not in emission_probs:
            emission_probs[tag] = {}
        for word, count in words.items():
            emission_probs[tag][word] = count / tag_count[tag]

    

    return []