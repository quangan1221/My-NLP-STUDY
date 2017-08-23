# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:31:55 2017

@author: qg186002
"""
################### NLP with python nltk toolkit study###############################33
#1. Tokenization
from nltk.tokenize import sent_tokenize, word_tokenize

#lexicon and corporas
#corpora - body og text/ ex: medical journals, presidental speeches
#lexison - words and their means


example_text='Hello Mr. Smith, how are you doing today?'

print(sent_tokenize(example_text))
print(word_tokenize(example_text))

import nltk
nltk.download()


#2.clean up data
#stop words
from nltk.corpus import stopwords

stop_words=set(stopwords.words("english"))
example_sentence="This is an example showing off stop word filteration."
words=word_tokenize(example_sentence)
filtered_sentence=[w for w in words if not w in stop_words]


#3. Stemming data
# I was taking a ride in the car.
# I was riding in the car. 
from nltk.stem import PorterStemmer

ps=PorterStemmer()
example_words=["python","pythoner","pythoning","pythoned","initialy","lying"]

for w in example_words:
    print (ps.stem(w))
    
    
new_text="It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at once."

words=word_tokenize(new_text)
for w in words:
    print(ps.stem(w))
    

#############4. Part of Speech Tagging####################
"""
Part of Speech tagging does exactly what it sounds like, it tags each word in a sentence with the part of speech for that word. 
This means it labels words as noun, adjective, verb, etc.
 PoS tagging also covers tenses of the parts of speech. 
 
 """
from nltk.corpus import inaugural
from nltk.tokenize import PunktSentenceTokenizer  #supervised  pre-trained

'''
POS tag list:
CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
'''



####inaugural 自带数据
train_text=inaugural.raw('1789-Washington.txt')
sample_text=inaugural.raw('2009-Obama.txt')

#sentence tokenization
custom_sent_tokenizer=PunktSentenceTokenizer(train_text)
tokenized=custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))
        
        
process_content()





############5. Chunck#########################

#Chunking in Natural Language Processing (NLP) is the process by which we group various words 
#together by their part of speech tags. 
"""
Regexp special characters

r"." matches any character
r"^…" / r"…$" matches the start / end of the string
r"…?" / r"…*" / r"…+" matches a regexp at most once / any number of times / at least once
r"…??" / r"…*?" / r"…+?" are the non-greedy alternatives
r"…|…|…" matches either this or that or that
r"[…]" matches any of the characters given
r"[^…]" matches any character not given
r"[^a-z]" matches any character between "a" and "z" (according to the unicode order)
r"[…-]" matches a minus (in addition to the other given characters)
r"[[…]" matches a closing bracket (in addition to the other given characters)
backslashed characters
r"\." / r"\[" / r"\\" / etc. matches the literal symbol 
r"\s" matches a whitespace character
r"\S" matches non-whitespace
r"\w" matches a letter or a digit or underscore 
r"\W" matches non-(letter|digit|underscore)
r"\d" matches a digit
r"\D" matches non-digit
r"\b" matches a word boundary
i.e., the empty string but only in the context of r"\W\w" or r"\w\W"
r"[^\W\d_]" matches only letters
try to understand why
the idea is taken from http://stackoverflow.com/questions/1673749
r"…(…)…" matches the whole regexp, but captures the part inside parenthesis so that you can look it up later
r"…(?:…)…" matches the regexp, and does not capture the parenthesis
"""
def process_content():

    words=nltk.word_tokenize(tokenized[1])
    tagged=nltk.pos_tag(words)
            
    #find all adverb, using regular expression
    #any forms of adverb,one more times
    chunkGram=r"""Chunk:{<RB.?>*<VB.?>*<NNP>+<NN>?}"""
    chunkParser=nltk.RegexpParser(chunkGram)
    chunked=chunkParser.parse(tagged)
    print(chunked)
    chunked.draw()
            
            
 ##Another example
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import Tree
 

tagged_text = "[ The/DT cat/NN ] sat/VBD on/IN [ the/DT mat/NN ] [ the/DT dog/NN ] chewed/VBD ./."
gold_chunked_text = tagstr2tree(tagged_text)
unchunked_text = gold_chunked_text.flatten()

#Chunking uses a special regexp syntax for rules that delimit the chunks. 
#These rules must be converted to 'regular' regular expressions before a sentence can be chunked.
tag_pattern="<DT>?<JJ>*<NN.*>"
regxp_pattern=tag_pattern2re_pattern(tag_pattern)

chunk_rule = ChunkRule("<.*>+", "Chunk everything")














##############Chinking###########################

#remove something from chunks
"""
Chinking is a part of the chunking process with natural language processing with NLTK. 
A chink is what we wish to remove from the chunk.
 We define a chink in a very similar fashion compared to how we defined the chunk. 
 """
import nltk

words=nltk.word_tokenize(tokenized[1])
tagged=nltk.pos_tag(words)
            
#chunck everything --keep everything together  全部归在一支下了
#Chink on verbs/prepositions/DETEMINER
chunkGram=r"""Chunk:{<.*>+}
                        }<VB.?|IN|DT|TO>+{"""   
chunkParser=nltk.RegexpParser(chunkGram)
chunked=chunkParser.parse(tagged)
print(chunked)
chunked.draw()


##################### Named Entity Recognition ##########################3
"""
Find subjects
Named entity recognition is useful to quickly find out what the subjects of discussion are. NLTK comes packed full of options for us. 
We can find just about any named entity, or we can look for specific ones.
""""

"""""
NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, MidlothianPerson
"""

##bad true positive rate


words = nltk.word_tokenize(5)
tagged = nltk.pos_tag(words)
namedEnt = nltk.ne_chunk(tagged)
namedEnt.draw()


#################corpus #########################
"""
The NLTK corpus is a massive dump of all kinds of natural language data sets that are definitely worth taking a look at.

Almost all of the files in the NLTK corpus follow the same rules for accessing them by using the NLTK module, but nothing is magical about them. 
These files are plain text files for the most part, some are XML and 
some are other formats, but they are all accessible by you manually, or via the module and Python. Let's talk about viewing them manually.


Path: %appdata% nltk_data
"""

from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg

# sample text
sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

for x in range(5):
    print(tok[x])
    
    
    
#################### wordnet #############################3
# lexicon : dictionary
"""
Part of the NLTK Corpora is WordNet. I wouldn't totally classify WordNet as a Corpora, if anything 
it is really a giant Lexicon, but, either way, it is super useful. With WordNet we can do things
 like look up words and their meaning according to 
their parts of speech, we can find synonyms, antonyms, and even examples of the word in use. 

"""

from nltk.corpus import wordnet

syns=wordnet.synsets("programs")  

#synset  同义词
print(syns)
print(syns[4].name())
#lemmas 词元 base of the word
print(syns[0].lemmas())
.
#just the word
print(syns[5].lemmas()[0].name())

#definition
print(syns[0].definition())

#examples
print(syns[0].examples())


# find synonyms and antonyms of good
synonyms=[]
antonyms=[]

for syn in wordnet.synsets("good"):
    #print(syn)
    for l in syn.lemmas():
        print(l)
        synonyms.append(l.name())
        if l.antonyms():
            #print (l.antonyms())
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))    
print(set(antonyms))      

# let's find semantic similarity between two words
#可用于查重#


w1=wordnet.synset("ship.n.01")

w2=wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2))



w3=wordnet.synset("car.n.01")

print(w1.wup_similarity(w3))

w4=wordnet.synset("cat.n.01")
w5=wordnet.synset("dog.n.01")

print(w3.wup_similarity(w4))

print(w5.wup_similarity(w4))





###################### Text Classification ##########################


from nltk.corpus import movie_reviews   #already labeled
import random
import nltk

#category: pos neg    get words vectors of every file in movie reviews
documents=[(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)] #tuples

#words: features  makes up elements

random.shuffle(documents)  #洗牌

all_words=[]
for w in movie_reviews.words():
    all_words.append(w.lower())
    
    
# nltk words frequency distribution
#39768 in total
allwords=nltk.FreqDist(all_words)
print(allwords.most_common(15))

#253 stupids in 1000 neg reviews
print(allwords["stupid"])


#######word feature learning ############

#排名前3000多的词
word_features=list(allwords.keys())[:5000]


def find_features(document):
    words=set(document)
    features={}  #dictionary
    for w in word_features:
        features[w]=(w in words)   #tell us whether the word is in top 3000 dictionary
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

#word as key, t/f as value
featuresets=[(find_features(rev),category) for (rev,category) in documents]      


##### Naive Bayes Classification##########
training_set=featuresets[:1900]
testing_set=featuresets[1900:]


#posterir = prior occurrencecs * likelihood/evidence 
classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo Accuracy:", nltk.classify.accuracy(classifier,testing_set))

classifier.show_most_informative_features(30)
"""
Most Informative Features
                 insipid = True              neg : pos    =     10.3 : 1.0
                   sucks = True              neg : pos    =     10.2 : 1.0
                   damon = True              pos : neg    =      7.9 : 1.0
                  spaces = True              pos : neg    =      7.0 : 1.0
               balancing = True              pos : neg    =      7.0 : 1.0
                republic = True              pos : neg    =      6.3 : 1.0
                    jude = True              pos : neg    =      6.3 : 1.0
                    noah = True              pos : neg    =      6.3 : 1.0
                religion = True              pos : neg    =      5.9 : 1.0
            manipulation = True              pos : neg    =      5.8 : 1.0
                balanced = True              pos : neg    =      5.7 : 1.0
                  venice = True              pos : neg    =      5.7 : 1.0
                 bronson = True              neg : pos    =      5.7 : 1.0
              scratching = True              neg : pos    =      5.7 : 1.0
                   vader = True              pos : neg    =      5.4 : 1.0
                hamilton = True              pos : neg    =      5.0 : 1.0
                  fuller = True              pos : neg    =      5.0 : 1.0
                 topping = True              pos : neg    =      5.0 : 1.0
                 redford = True              pos : neg    =      5.0 : 1.0
                 gangsta = True              pos : neg    =      5.0 : 1.0
"""


classifier.classify(testing_set[19][0])


a=testing_set[0][0]
a.keys()[a.values()]
[name for name, boole in a.items() if boole == True]
print(testing_set[0][0](testing_set[0][0].values()==True)