import io, sys
import numpy as np
from heapq import *

def load_vectors(filename):
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))
    return data

## This function computes the cosine similarity between vectors u and v

def cosine(u, v):
    ## FILL CODE
    cos = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
    return cos

## This function returns the word corresponding to 
## nearest neighbor vector of x
## The list exclude_words can be used to exclude some
## words from the nearest neighbors search

def nearest_neighbor(x, word_vectors, exclude_words=[]):
    best_score = -1.0
    best_word = None

    ## FILL CODE
    for i in word_vectors:
        if i not in exclude_words and not (x ==  word_vectors[i]).all():
            dist = cosine(x, word_vectors[i])
            if dist > best_score:
                best_score = dist
                best_word = i

    return best_word

## This function return the words corresponding to the
## K nearest neighbors of vector x.
## You can use the functions heappush and heappop.

def knn(x, vectors, k):
    heap = []

    ## FILL CODE
    for i in vectors:
        if len(heap)>k:
            heappush(heap, (cosine(x, vectors[i]),i))
            heappop(heap)
        
        else:
            heappush(heap, (cosine(x,vectors[i]), i))

    return [heappop(heap) for i in range(len(heap))][::-1][1:]

## This function return the word d, such that a:b and c:d
## verifies the same relation

def analogy(a, b, c, word_vectors):
    ## FILL CODE
    a = a.lower()
    b = b.lower()
    c = c.lower()
    x_a, x_b, x_c = word_vectors[a], word_vectors[b], word_vectors[c]
    x_a = x_a/np.linalg.norm(x_a)
    x_b = x_b/np.linalg.norm(x_b)
    x_c = x_c/np.linalg.norm(x_c)
    
    best_score = float('-inf')
    best_word = ''
    for i in word_vectors.keys():
#         if True in [w in i for w in [a, b, c]]:
        if i in [a, b, c]:
            continue
            
        v = word_vectors[i]/np.linalg.norm(word_vectors[i])
        d = (x_b - x_a + x_c ).dot(v)
        if d > best_score:
            best_score = d
            best_word = i
            
    return best_word
    

## Compute the association strength between:
##   - a word w
##   - two sets of attributes A and B

def association_strength(w, A, B, vectors):
    strength = 0.0
    ## FILL CODE
    s1 = 0.0
    s2 = 0.0
    for i in A:
        s1 +=cosine(vectors[w], vectors[i])
    #print(s1)
    for j in B:
        s2 += cosine(vectors[w], vectors[j])
    
    #print(s2)
    strength = (1/len(A))*s1 - (1/len(B)*s2)
    return strength

## Perform the word embedding association test between:
##   - two sets of words X and Y
##   - two sets of attributes A and B

def weat(X, Y, A, B, vectors):
    score = 0.0
    ## FILL CODE
    score_1 = 0.0
    score_2 = 0.0
    for x in X:
        score_1 += association_strength(x, A,B, vectors)
    
    for y in Y:
        score_2 += association_strength(y, A,B, vectors)
    score = score_1 - score_2
    return score

######## MAIN ########

print('')
print(' ** Word vectors ** ')
print('')

word_vectors = load_vectors(sys.argv[1])

print('similarity(apple, apples) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['apples']))
print('similarity(apple, banana) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['banana']))
print('similarity(apple, tiger) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['tiger']))

print('')
print('The nearest neighbor of cat is: ' +
      nearest_neighbor(word_vectors['cat'], word_vectors))

knn_cat = knn(word_vectors['cat'], word_vectors, 5)
print('')
print('cat')
print('--------------')
for score, word in knn(word_vectors['cat'], word_vectors, 5):
    print (word + '\t%.3f' % score)

print('')
print('france - paris + rome = ' + analogy('paris', 'france', 'rome', word_vectors))

## A word about biases in word vectors:

print('')
print('similarity(genius, man) = %.3f' %
      cosine(word_vectors['man'], word_vectors['genius']))
print('similarity(genius, woman) = %.3f' %
      cosine(word_vectors['woman'], word_vectors['genius']))

## Replicate one of the experiments from:
##
## Semantics derived automatically from language corpora contain human-like biases
## Caliskan, Bryson, Narayanan (2017)

career = ['executive', 'management', 'professional', 'corporation', 
          'salary', 'office', 'business', 'career']
family = ['home', 'parents', 'children', 'family',
          'cousins', 'marriage', 'wedding', 'relatives']
male = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
female = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']

print('')
print('Word embedding association test: %.3f' %
      weat(career, family, male, female, word_vectors))
