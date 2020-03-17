import io, sys, math, re
from collections import defaultdict
import numpy as np

# GOAL: build a stupid backoff bigram model

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


def remove_rare_words(data, vocab, mincount):
    ## FILL CODE
    # replace words in data that are not in the vocab 
    # or have a count that is below mincount
    data_with_unk = data[:]
    for w1, sentence  in enumerate(data):
        for w2, word in enumerate(sentence):
            if vocab[word]<mincount or word not in vocab:
                data_with_unk[w1][w2] = '<unk>'
            
    
    return data_with_unk


def build_bigram(data):
    unigram_counts = defaultdict(lambda:0)
    bigram_counts  = defaultdict(lambda: defaultdict(lambda: 0.0))
    total_number_words = 0

    ## FILL CODE
    # Store the unigram and bigram counts as well as the total 
    # number of words in the dataset
    

    for sentence in data:
        for i, word in enumerate(sentence):
            unigram_counts[word] += 1
            total_number_words += 1
            
            if i <len(sentence)-1:
            
                bigram_counts[word][sentence[i+1]] +=1
        

    unigram_prob = defaultdict(lambda:0)
    bigram_prob = defaultdict(lambda: defaultdict(lambda: 0.0))

    ## FILL CODE
    # Build unigram and bigram probabilities from counts
    
    for  sentence in data:
        for i, word  in enumerate(sentence):
            unigram_prob[word] = (1.*unigram_counts[word])/ total_number_words
            if i <len(sentence)-1:
                
                bigram_prob [word][sentence[i+1]] = (1.*bigram_counts[word][sentence[i+1]])/ unigram_counts[word]
                                 

    return {'bigram': bigram_prob, 'unigram': unigram_prob}

def get_prob(model, w1, w2):
    assert model["unigram"][w2] != 0, "Out of Vocabulary word!"
    ## FILL CODE
    # Should return the probability of the bigram (w1w2) if it exists
    # Else it return the probility of unigram (w2) multiply by 0.4
    if  model['bigram'][w1][w2]!=0:
        return model['bigram'][w1][w2]
    else:
        return 0.4*model['unigram'][w2]

def perplexity(model, data):
    ## FILL CODE
    # follow the formula in the slides
    # call the function get_prob to get P(w2 | w1)
    perp = 0.0
    for sentence in data:
        prob = 0.0
        for i in range(len(sentence)-1):
            prob += np.log(get_prob(model, sentence[i], sentence[i+1]))
        
        prob = -prob/len(sentence)
        perp += prob
    perplex = perp/len(data)                    
        
    return np.exp(perplex)

def generate(model):
    sentence = ["<s>"]
    ## FILL CODE
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    for i in range(10):
        if sentence[-1]=='</s>':
            break
        x = list(model['bigram'][sentence[-1]].keys())
        y = list(model['bigram'][sentence[-1]].values())
        word = np.random.choice(x,1, p=y)
        sentence.append(word[0])
    if sentence[-1] !='</s>':
        sentence.append('</s>')
    return sentence

###### MAIN #######

print("load training set")
train_data, vocab = load_data("train2.txt")
## FILL CODE 
# If you have a Out of Vocabulary error (OOV) 
# call the function "remove_rare_words" to replace 
# rare words with <unk> in the dataset
train_data = remove_rare_words(train_data, vocab, mincount=10)
print("build bigram model")
model = build_bigram(train_data)

print("load validation set")
valid_data, _ = load_data("valid2.txt")
## FILL CODE 
# If you have a Out of Vocabulary error (OOV) 
# call the function "remove_rare_words" to replace 
# OOV with <unk> in the dataset
valid_data = remove_rare_words(valid_data, vocab, mincount=10)
print("The perplexity is", perplexity(model, valid_data))

print("Generated sentence: ",generate(model))
