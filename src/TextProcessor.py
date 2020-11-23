import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer 
import re

class TextProcessor():

    def __init__(self):
        """ Baseline text processor"""

    def process(self,text): #Change this for the second task
        """ Input: List of words [w1,w2,w3,....]"""
        """ Output: List of token (allows to split words into parts:) """
        
        stemmer = PorterStemmer()
        #lemmatizer = WordNetLemmatizer()

        text = [stemmer.stem(word) for word in text]
        #text = [lemmatizer.lemmatize(word) for word in text]

        #result += re.findall(r"[\w']+|[.,!?]",word)]

        return text

