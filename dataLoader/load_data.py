
import dataset
import os

import sys
sys.path.insert(0, '../gloveLoader/')
from loadGlove import loadGloveModel, getKeyedVect

def load_data(dir,num=10):
    #file = open(filename, 'r') 
    #print(file.readline())
    sents = dataset.loopFiles(dir,num)
    return sents


if __name__=='__main__':

    #model = getKeyedVect('../gloveloader/glove_50d_keyed.txt')
    sentences = load_data('./newsfiles/newsfiles/')
    for s in sentences:
        print(s)
    
