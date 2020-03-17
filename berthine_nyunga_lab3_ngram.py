import io, sys, math, re
from collections import defaultdict
import numpy as np

# GOAL: build a stupid backoff ngram model

def load_data(filename):
    fin = io.open(filename, 'r', encoding='utf-8')
    data = []
    vocab = defaultdict(lambda:0)
    for line in fin:
        sentence = line.split()
        data.append(sentence)
        for word in sentence:
            vocab[word] += 1
    return data, vocab

def remove_rare_words(data, vocab, mincount=0):
    ## FILL CODE
    # replace words in data that are not in the vocab 
    # or have a count that is below mincount
    data_with_unk = data[:]
    for w1, sentence  in enumerate(data):
        for w2, word in enumerate(sentence):
            if vocab[word] < mincount or word not in vocab :
                data_with_unk[w1][w2] = '<unk>'
            
    
    return data_with_unk


def build_ngram(data, n):
    total_number_words = 0
    counts = defaultdict(lambda: defaultdict(lambda: 0.0))

    for sentence in data:
        sentence = tuple(sentence)
        ## FILL CODE
        # dict can be indexed by tuples
        # store in the same dict all the ngrams
        # by using the context as a key and the word as a value
        for j in range(n):
            for i in range(len(sentence)-j):
                total_number_words +=1
                counts[sentence[i:(i+j)]][sentence[i+j]] +=1
    total_number_words = total_number_words / n

    prob  = defaultdict(lambda: defaultdict(lambda: 0.0))
    ## FILL CODE
    # Build the probabilities from the counts
    # Be careful with how you normalize!
    for  word  in counts:
        for w in counts[word]:
            prob[word][w] = counts[word][w]/sum(counts[word].values())

    return prob


def get_prob(model, context, w):
    ## FILL CODE
    # code a recursive function over 
    # smaller and smaller context
    # to compute the backoff model
    # Bonus: You can also code an interpolation model this way
    if model[context][w] !=0:
        return model[context][w]
    else :
        return 0.4*get_prob(model, context[1:],w)
    

def perplexity(model, data, n):
    ## FILL CODE
    # Same as bigram.py
    perp = 0
    for sentence in data:
        sentence = tuple(sentence)
        prob = 0
        for i in range(1, len(sentence)):
            if i>= n-1:
                prob += np.log(get_prob(model, sentence[i-n+1:i], sentence[i]))
            else:
                prob += np.log(get_prob(model, sentence[:i], sentence[i]))
        prob = -prob/len(sentence)
        perp += prob
    
    return np.exp(perp/len(data))
    

def get_proba_distrib(model, context):
    ## FILL CODE
    # code a recursive function over context
    # to find the longest available ngram 
    if context  in model:
        return context
    else:
        return get_proba_distrib(model, context[1:])

def generate(model):
    sentence = ["<s>"]
    ## FILL CODE
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    sentence = ['<s>']
    for length_sentence in range(20):
        if sentence[-1]=='</s>':
            break
        x= list(model[get_proba_distrib(model, tuple(sentence))].keys())
        y= list(model[get_proba_distrib(model, tuple(sentence))].values())
        word = np.random.choice(x, 1, p=y)
        sentence.append(word[0])
    if sentence[-1] !='</s>':
        sentence.append('</s>')
    return sentence

###### MAIN #######

n = 2

print("load training set")
train_data, vocab = load_data("train.txt")

## FILL CODE
# Same as bigram.py
train_data = remove_rare_words(train_data, vocab, mincount=4)

print("build ngram model with n = ", n)
model = build_ngram(train_data, n)

print("load validation set")
valid_data, _ = load_data("valid.txt")
## FILL CODE
# Same as bigram.py
valid_data = remove_rare_words(valid_data, vocab, mincount=4)

print("The perplexity is", perplexity(model, valid_data, n))

print("Generated sentence: ",generate(model))

