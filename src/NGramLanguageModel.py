import math
from collections import defaultdict
from LanguageModel import LanguageModel


class NGramLanguageModel(LanguageModel):

    #Default
    word_counts = defaultdict(lambda: defaultdict(lambda: 0)) #instantiating the data structure that will hold the counts
    bigrams = defaultdict(lambda: defaultdict(lambda: 0))
    unigrams = defaultdict(lambda: defaultdict(lambda: 0))
    vocabulary_size = 0 #total words
    N = 0 #will store word count

    def __init__(self,size): 
        super().__init__(size)
        self.size = size
        
        
    #Returns a list as a string
    def list_to_string(self,sub_list):  
        sequence = ""  
        for element in sub_list:
            if len(sub_list) == 1:
                sequence += element
            else: 
                sequence += element
                sequence+= ' '

        return sequence.strip() 


    def train(self,tokens):
        """ Train the language model on the list of tokens where tokens is a list of all the tokens (including some punctuation marks). """

        NGramLanguageModel.vocabulary_size = len(set(tokens))
        NGramLanguageModel.N = len(tokens)
        #insert a start and of sentence <s> ,</s> symbols
        text = ["<s>"]*(self.size-1) + tokens + ["</s>"]*(self.size-1)
 
        #Populating data structure in this formatL P(W|H) --> {H:{W:count}}  P(W)--unigram--> {'':{W:count}}  #tuple(token[i:i+(self.size-1)])
        for i in range(len(tokens)):

            history = self.list_to_string(text[i:i+(self.size-1)])
            nth_word = text[i+(self.size-1)]
            NGramLanguageModel.word_counts[history][nth_word] +=1 #If the history exists, a new word is added if that word exists it is incremented, otherwise a new histroy is added with following word
    
        #For BackOff and Interpolation:
        #calculating the bigrams
        for i in range(len(tokens)):

            history = self.list_to_string(text[i:i+(2-1)])
            nth_word = text[i+(2-1)]

            NGramLanguageModel.bigrams[history][nth_word] +=1 #I
        #calculating the unigrams
        for i in range(len(tokens)):
        
            nth_word = text[i]
            NGramLanguageModel.unigrams[''][nth_word] +=1 #I

        #Testing stuff
        #  print(word_counts)
        # for k,v in word_counts.items():
        #     print('history',k)
        #     for w,count in v.items():
        #         print("word:",w)
        #         print("count:",count)

        return NGramLanguageModel.word_counts , NGramLanguageModel.bigrams, NGramLanguageModel.unigrams 


    def discount(self,count):
    
        discounted_count = 0.0
        discounted_count = (count+1)*(NGramLanguageModel.N/(NGramLanguageModel.N+NGramLanguageModel.vocabulary_size))

        return discounted_count


    def back_off_smoothing(self, ngram, h, w):

        smoothed_probability = 0.0
        counts = NGramLanguageModel.word_counts[h][w]

        #back off    #MAYBE THIS COULD BE DONE RECURSIVELY?
        if counts == 0:

            h = self.list_to_string(ngram[1:-1])
            w = ngram[-1]
            counts = NGramLanguageModel.bigrams[h][w]
            #further backoff
            if counts == 0:
                h = ''
                w = ngram[-1]
                counts = NGramLanguageModel.unigrams[h][w]
                total_occurances = sum(NGramLanguageModel.unigrams[h].values())
            else:
                total_occurances = sum(NGramLanguageModel.bigrams[h].values())
        else:
            total_occurances = sum(NGramLanguageModel.word_counts[h].values())

        discounted_count = self.discount(counts)
        smoothed_probability = discounted_count/total_occurances 

        return smoothed_probability


    def laplace_smoothing(self, counts, total_occurances):

        #Unknown word
        if total_occurances == 0:
            laplace_smoothing = ((counts + 1)/(NGramLanguageModel.N + NGramLanguageModel.vocabulary_size)) 
        else:
            laplace_smoothing = ((counts + 1)/(total_occurances + NGramLanguageModel.vocabulary_size)) 
            #laplace_smoothing = ((counts)/(total_occurances)) 

        return laplace_smoothing


    def linear_interpolation(self,ngram, h, w):
    
        lambda1 = 0.3
        lambda2 = 0.5
        lambda3 = 0.2

        counts = NGramLanguageModel.word_counts[h][w]
        total_occurances = sum(NGramLanguageModel.word_counts[h].values())
        tri_prob =  self.laplace_smoothing(counts,total_occurances)


        h = self.list_to_string(ngram[1:-1])
        w = ngram[-1]
        counts = NGramLanguageModel.bigrams[h][w]
        total_occurances = sum(NGramLanguageModel.bigrams[h].values())
        bi_prob = self.laplace_smoothing(counts,total_occurances)


        h = ''
        w = ngram[-1]
        counts = NGramLanguageModel.unigrams[h][w]
        total_occurances = sum(NGramLanguageModel.unigrams[h].values())
        uni_prob = self.laplace_smoothing(counts,total_occurances)

        interpolated_prob = (lambda1 * uni_prob) + (lambda2 * bi_prob) + (lambda3 * tri_prob)

        return interpolated_prob


    def calcLogProb(self,ngram):
        """ Calculate probability of the ngram list of tokens p(ngram[-1]|ngram[:-1]))"""
        """ From getPPL this method receives a given an ngram i.e. a given history (n-1 words) which """

        log_probability = 0.0 

        #Split incoming ngram into history and word
        h = self.list_to_string(ngram[:(self.size-1)])
        w = ngram[(self.size-1)]

        counts = NGramLanguageModel.word_counts[h][w]# returns count of w given h. If h unknown then returns 0
        total_occurances = sum(NGramLanguageModel.word_counts[h].values())
        
        #laplace_probability = self.laplace_smoothing(counts,total_occurances) 

        #Different smoothing methods: 
        #print(laplace_probability)
        #log_probability = math.log2(laplace_probability)

        # discounted_probability = self.back_off_smoothing(ngram,h,w)        
        # log_probability = math.log2(discounted_probability)

        interpolated_probability = self.linear_interpolation(ngram,h,w)
        log_probability = math.log2(interpolated_probability)

        return log_probability