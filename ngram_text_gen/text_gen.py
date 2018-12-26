
from tqdm import tqdm





bigram_test = './bigram.txt'
bigram_most_likely = {}
bigram_tracker = {}
bigram_pairs = {}

FREQ_CUTOFF = 100


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


sent = 'car drove hill fast'

#print(bigram_pairs[('hello','there')])
#print(bigram_most_likely['hello'])

ind1 = 0
ind2 = 1

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
        print('Is',split[i],'in the bigram set?')


        if(split[i] in bigram_most_likely.keys()):
            print('yes')
            if(dict_ind>=len(bigram_most_likely[split[i]])):
                flag = False
                continue

            if(i+1<len(split)):
                if(bigram_most_likely[split[i]][dict_ind] in bigram_most_likely.keys()):
                    if(split[i+1] in bigram_most_likely[bigram_most_likely[split[i]][dict_ind]]):
                        print('next word:',split[i+1],'is in',bigram_most_likely[split[i]][dict_ind])
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









