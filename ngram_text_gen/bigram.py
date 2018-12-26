from tqdm import tqdm


def smooth_sentences(sents,frequency_cutoff=100, bigram_file='./bigram.txt'):


    bigram_test = bigram_file
    bigram_most_likely = {}
    bigram_tracker = {}
    bigram_pairs = {}

    FREQ_CUTOFF = frequency_cutoff


    f = open(bigram_test,'r',encoding="ISO-8859-1")
    ind = 0
    print('Loading bigrams...')
    for line in f:
        split = line.split()

        w1 = split[1]
        w2 = split[2]
        freq = split[0]


        bigram_pairs[(split[1],split[2])] = split[0]
        if split[1] not in bigram_tracker:
            bigram_tracker[w1] = freq

        
        if(int(freq)>=FREQ_CUTOFF):
            if w1 not in bigram_most_likely.keys():
                bigram_most_likely[w1] = list()
            bigram_most_likely[w1].append(w2)


        ind+=1
    print('Loaded',ind,'bigrams')


    for sent in sents:

        new_sent = ""
        split = sent.split()
        for i in range(len(split)):
            dict_ind = 0
            flag = True
            
            new_sent+=split[i]+' '
            if(i+1<len(split)):
                if(split[i] in bigram_most_likely.keys()):
                    if(split[i+1] in bigram_most_likely[split[i]]):
                        continue
            while(flag):


                if(split[i] in bigram_most_likely.keys()):
                    if(dict_ind>=len(bigram_most_likely[split[i]])):
                        flag = False
                        continue

                    if(i+1<len(split)):
                        if(bigram_most_likely[split[i]][dict_ind] in bigram_most_likely.keys()):
                            if(split[i+1] in bigram_most_likely[bigram_most_likely[split[i]][dict_ind]]):
                                new_sent+=bigram_most_likely[split[i]][dict_ind]+ ' '
                                flag = False
                                continue
                        dict_ind+=1
                        continue
                    else:
                        flag=False
                else:
                    flag=False

        print(new_sent)
        print(bigram_pairs[('newspaper','the')])



if __name__ == '__main__':
    sent1 = ['1989 Jefferson Portland blazers games considering NBA chance shortfall Cardinals attendance fans quickly seasons serving']
    sent2 = 'newspaper website technology'

    sents = sent1
    smooth_sentences(sents)





