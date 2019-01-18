#!/bin/python
import nltk
import os
import json
import  sys, os, re

lexicon_dict = dict()
# common_words = []
common_words = nltk.corpus.stopwords.words('english')
location = set()
cluster = dict()


def load_json(filename):
    with open(filename, 'r') as myfile:
        # print(type(myfile))
        return json.load(myfile)


def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    global common_words
    global cluster

    data = load_json('300D_100000-words_1000-clusters.json')
    data2 = load_json('300D_68283-words_50-clusters.json')
    for each_list in data2:
        for each_word in each_list:
            cluster[each_word] = data2.index(each_list)

    rootDir = 'data/lexicon'
    # print str(r)
    skip_files_stop_words = ['lower.100', 'lower.500', 'lower.1000', 'lower.5000', 'lower.10000', 'english.stop']
    skip_files_loc = ['location', 'location.country']
    for dirName, subdirList, fileList in os.walk(rootDir):
        for each in fileList:
            if each not in skip_files_stop_words:
                fname = 'data/lexicon/' + each
                # print(fname)
                lexicon_dict[each] = set()
                for line in open(fname):
                    line = line.strip().split()
                    # line = line.split()
                    for word in line:
                        # word = line.split(" ")
                        lexicon_dict[each].add(word)
    filename = 'data/lexicon/lower.10000'
    for line in filename:
        line = unicode(line.strip())
        common_words.append(line)
    filename = 'data/lexicon/english.stop'
    for line in filename:
        line = unicode(line.strip())
        common_words.append(line)
    common_words = set(common_words)

    # filenamel = 'data/lexicon/location'
    # for line in filenamel:
    #     line = unicode(line.strip())
    #     location.add(line)
    # filenamel = 'data/lexicon/location.country'
    # for line in filenamel:
    #     line = unicode(line.strip())
    #     location.add(line)
    # return lexicon_dict, common_words
    pass


def token2features(sent, i, wordtag, add_neighs=True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent) - 1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    if word[0].isupper() and i != 0:
        ftrs.append("PNP")
    if word.startswith('@') or word.startswith('#'):
        ftrs.append("STARTS_WITH")
    if "http" or ".com" in word:
        ftrs.append("LINK")
    if word in cluster:
        ftrs.append("CLUSTER_IDX=" + str(cluster[word]))
    ftrs.append("WORD_LEN=" + str(len(word)))
    ftrs.append("WORD_TAG=" + wordtag[i][1])
    lexicon_list = []
    for k, vset in lexicon_dict.iteritems():
        if word in vset:
            lexicon_list.append(k)
            ftrs.append("LEXICON_OF_WORD=" + str(lexicon_list))
    # l = len(word)
    # if l > 7:
    #     l = 7
    # ngrams = [['char-' + str(n) + '-gram=' + word[j:j + n]
    #        for j in range(len(word) - n + 1)] for n in range(1, l)]
    # ngrams_flat = [ng for sublist in ngrams for ng in sublist]
    # print(ngrams)
    # print(ngrams_flat)
    # exit()
    # for gram in ngrams_flat:
    #     ftrs.append(gram)
    if word in common_words:
        ftrs.append("COMMON_WORD")

    # # elif word in location:
    # #     ftrs.append("LOCATION")
    else:
        context_win2 = set()
        context_win3 = set()
        if i < len(sent) - 1:
            context_win2.add(word)
            context_win2.add(unicode(sent[i + 1]))
        if i < len(sent) - 2:
            context_win3.add(word)
            context_win3.add(unicode(sent[i + 1]))
            context_win3.add(unicode(sent[i + 2]))
        if add_neighs:
            for k, vset in lexicon_dict.iteritems():
                if word in vset:
                    ftrs.append("LEXICON_OF_WORD=" + k)
                if len(context_win2) > 0:
                    if context_win2.intersection(vset) == context_win2:
                        ftrs.append("LEXICON_OF_2=" + k)
                if len(context_win3) > 0:
                    if context_win2.intersection(vset) == context_win3:
                        ftrs.append("LEXICON_OF_3=" + k)

    # ftrs.append("IDX="+str(i))
    # if len(word) < 4:
    #     ftrs.append("word-len<4:" + str(len(word)))

    # previous/next word feats
    if (len(word) >= 4):
        ftrs.append("prefix=%s" % word[0:1].lower())
        ftrs.append("prefix=%s" % word[0:2].lower())
        ftrs.append("prefix=%s" % word[0:3].lower())
        ftrs.append("suffix=%s" % word[len(word) - 1:len(word)].lower())
        ftrs.append("suffix=%s" % word[len(word) - 2:len(word)].lower())
        ftrs.append("suffix=%s" % word[len(word) - 3:len(word)].lower())

    if re.search(r'^[A-Z]', word):
        ftrs.append('INITCAP')
    # if re.search(r'^[A-Z]', word) and goodCap:
    #     ftrs.append('INITCAP_AND_GOODCAP')
    if re.match(r'^[A-Z]+$', word):
        ftrs.append('ALLCAP')
    # if re.match(r'^[A-Z]+$', word) and goodCap:
    #     ftrs.append('ALLCAP_AND_GOODCAP')
    if re.match(r'.*[0-9].*', word):
        ftrs.append('HASDIGIT')
    if re.match(r'[0-9]', word):
        ftrs.append('SINGLEDIGIT')
    if re.match(r'[0-9][0-9]', word):
        ftrs.append('DOUBLEDIGIT')
    if re.match(r'.*-.*', word):
        ftrs.append('HASDASH')
    if re.match(r'[.,;:?!-+\'"]', word):
        ftrs.append('PUNCTUATION')
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i - 1, wordtag, add_neighs=False):
                ftrs.append("PREV_" + pf)
        if i < len(sent) - 1:
            for pf in token2features(sent, i + 1, wordtag, add_neighs=False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs


if __name__ == "__main__":
    sents = [
        ["I", "love", "food", "http://tinyurl.com/26zeju5"]
    ]

    for sent in sents:
        wordtag = nltk.pos_tag(sent)
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i, wordtag)
