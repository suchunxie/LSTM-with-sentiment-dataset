import re
from nltk.tokenize import RegexpTokenizer


def to_clean_sentence(sentence):
    #sentence=str(sentence)
    #print("sentence",sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    #tokenizer = RegexpTokenizer("[\w']+")
    tokens = tokenizer.tokenize(rem_num)  
    #filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    #filtered_words = [w for w in tokens if len(w) > 2]
    #stem_words=[stemmer.stem(w) for w in filtered_words]
    
    ##filtered_words = [w for w in tokens if len(w) > 0]
    #stem_words=[stemmer.stem(w) for w in filtered_words]
    #lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(tokens)

def to_clean_sentence_for_word2vector(sentence, vocab):
    #sentence=str(sentence)
    #print("sentence",sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    #tokenizer = RegexpTokenizer("[\w']+")
    tokens = tokenizer.tokenize(rem_num)  
    vocab.append(tokens)
    #filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    #filtered_words = [w for w in tokens if len(w) > 2]
    #stem_words=[stemmer.stem(w) for w in filtered_words]
    
    ##filtered_words = [w for w in tokens if len(w) > 0]
    #stem_words=[stemmer.stem(w) for w in filtered_words]
    #lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(tokens)