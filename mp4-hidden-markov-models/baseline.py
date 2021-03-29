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
# Modified Spring 2021 by Kiran Ramnath
from collections import Counter

"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    INVALID_TAGS = {'START', 'END'}

    counter_dict = {}
    tag_count = Counter() # just tag as key

    total_train_word_count = 0

    for sentence in train:
        for word, tag in sentence:
            if tag not in INVALID_TAGS:
                if word not in counter_dict:
                    counter_dict[word] = Counter()
                counter_dict[word].update({tag: 1})
                tag_count.update({tag: 1})

                total_train_word_count += 1

    most_common_tag = tag_count.most_common(1)[0][0]    

    output = []
    for sentence in test:
        output_sent = []
        for word in sentence:
            if word not in INVALID_TAGS:
                if word in counter_dict:
                    output_sent.append((word, counter_dict[word].most_common(1)[0][0])) # get most common tag for that word
                else:
                    output_sent.append((word, most_common_tag))
            else:
                output_sent.append((word, word)) # only for START and END
        output.append(output_sent)

    return output