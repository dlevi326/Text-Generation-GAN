import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.stem import WordNetLemmatizer

# Returns dict mapping words to its embedding

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded! With dimensions of:",len(model[list(model.keys())[0]]))
    print('Obtaining keyed vector')
    temp_file = gloveFile[:-4]+'_keyed.txt'
    glove2word2vec(gloveFile, temp_file)
    keyed_vect = KeyedVectors.load_word2vec_format(temp_file)
    print('Keyed vector obtained!')
    return model,keyed_vect

def getKeyedVect(filename):
    print('Obtaining keyed vector')
    return KeyedVectors.load_word2vec_format(filename)
    print('Keyed vector obtained!')






if __name__=='__main__':
    #model,gensim_model = loadGloveModel('./glove_50d.txt')
    gensim_model = getKeyedVect('./glove_50d_keyed.txt')
    zer = np.zeros(50)

    wordnet_lemmatizer = WordNetLemmatizer()
    sent = 'Hello my name is david and ive been walking many dogs a while'.split()
    new = map(wordnet_lemmatizer.lemmatize,sent)

    words = filter(lambda x: x in gensim_model.vocab, sent)

    new_sim_sent = ''
    #print(gensim_model.most_similar(positive='dog')[0][0])
    for w in words:
        new_sim_sent+=gensim_model.most_similar(positive=w)[0][0]
        new_sim_sent+=' '
    #print(new_sim_sent)
    #print(gensim_model.most_similar(positive=words))