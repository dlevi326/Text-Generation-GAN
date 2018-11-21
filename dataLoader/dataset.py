import json, random, re, os
from nltk import sent_tokenize

# Gets text from json file
def getText(filename):
    with open(filename, 'rb') as f:
        file = json.load(f)
    text = file['text']
    text = text.replace('\n','')  # removing newlines '\n' from text
    textlist = sent_tokenize(text)#list(map(str.strip, re.split(r"[.!?](?!$)", text)))  # split text into sentences
    # The first and last sentences were sometimes funky
    textlist = textlist[1:-1]
    if len(textlist) > 1:  # if there are more than one sentence in text
        return text, random.choice(textlist)
    else:
        return "None"

# Loops through folder and calls getText for each json file
def loopFiles(directory,num):
    tot_files = 0
    sents = []
    
    for ind,file in enumerate(os.listdir(directory)):
        if(ind>=num and num!=-1):
            break
        filename = directory + file
        filename = str(filename)

        if filename.endswith(".json"):
            with open(filename, 'rb') as f:
                file = json.load(f)
            text = file['text']
            text = text.replace('\n','')  # removing newlines '\n' from text
            textlist = sent_tokenize(text)#list(map(str.strip, re.split(r"[.!?](?!$)", text)))  # split text into sentences
            # The first and last sentences were sometimes funky
            textlist = textlist[1:-1]
            for t in textlist:
                sents.append(t)
            #sents.append(getText(filename))
            tot_files+=1
    print("Loaded",tot_files,"files")
    return sents


#data_folder = (os.getcwd() + "\\newsfiles\\")

#loopFiles(data_folder)
