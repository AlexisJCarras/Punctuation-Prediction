import math

class PunctuationPredictor:

    def __init__(self,lm):
        self.lm = lm
        self.punc = [",",".","!","?"]
        self.weights = [0.2]*(len(self.punc)+1) #to include for now punctuation

    def loadData(self,text):
        text,label = self.createLabelData(text)
        before = math.ceil(self.lm.getSize()/2)
        after = math.floor(self.lm.getSize()/2)
        text = ["<s>"] * before + text + ["</s>"] * after
        allProbs = []
        labelCount=[0]*len(self.weights)
        for i in range(len(text)-self.lm.getSize()):
            ngram = text[i+1:i+self.lm.getSize()+1]
            probs = []
            probs.append(self.lm.calcLogProb(ngram))

            c=0
            for j in range(len(self.punc)):
                n = ngram[1:before]+[self.punc[j]]+ngram[-1*after:]
                probs.append(self.lm.calcLogProb(n))
                if label[i] == self.punc[j]:
                    c = j+1
            if(i < 0):
                print (ngram,label[i],c,probs)
            #    exit()
            labelCount[c] += 1


            allProbs.append([probs,c])
        return allProbs,labelCount

    def optThresholds(self,text):
        allProbs,labelCount = self.loadData(text)

        """ Optimize weights"""
        for iter in range(10):
            self.findWeights(allProbs)
        print ("Best weights: ",self.weights)
        correct, correctLabels,metrics = self.calcScore(allProbs)
        print ("Correct: ",correct)
        print (metrics)
        print ("Correct Non Punct",correctLabels[0],labelCount[0])
        for i in range(len(self.punc)):
            print ("Correct ",self.punc[i],": ",correctLabels[i+1],labelCount[i+1])

    def addPuncuation(self,text):
        allProbs,labelCount = self.loadData(text)

        correct, correctLabels,metrics = self.calcScore(allProbs)
        print ("Correct: ",correct)
        print (metrics)
        print ("Correct Non Punct",correctLabels[0]," of ",labelCount[0])
        for i in range(len(self.punc)):
            print ("Correct ",self.punc[i],": ",correctLabels[i+1]," of ",labelCount[i+1])



    def findWeights(self,prob):
        for dim in range(len(self.weights)):
            best = -1
            bestscore = -1
            for f in [0.01,0.05,0.1,0.2,0.5,0.75,1]:
                self.weights[dim] = f
                correct,_,metrics = self.calcScore(prob)
                if metrics["f-score"] > bestscore:
                    bestscore = metrics["f-score"]
                    best = f
            self.weights[dim] = best

    def calcScore(self,prob):
        correct = 0
        correctLabels=[0]*len(self.weights)
        punc=0
        puncPred=0
        puncCorr=0
        count = 0
        for p, c in prob:
            pred = [i[0] * i[1] for i in zip(p, self.weights)]
            m = max(range(len(pred)), key=pred.__getitem__)
            if m == c:
                correct += 1
                if (m != 0):
                    puncCorr += 1
                correctLabels[m] += 1
            if(c != 0):
                punc += 1
            if (m != 0):
                puncPred += 1
            count += 1
            #if (count < 20):
            #    print (p,c,m)
        metrics = {}
        metrics["recall"] = puncCorr/punc
        if(puncPred == 0):
            metrics["precision"] = 0
        else:
            metrics["precision"] = puncCorr/puncPred
        if(metrics["recall"]+ metrics["precision"] == 0):
            metrics["f-score"] = 0
        else:
            metrics["f-score"] = 2*metrics["recall"] * metrics["precision"]/(metrics["recall"]+ metrics["precision"])
        return correct,correctLabels,metrics

    def createLabelData(self,text):
        t = []
        l = []
        for i in range(len(text)):
            if(text[i] in self.punc):
                l[-1] = text[i]
            else:
                t.append(text[i])
                l.append("0")
        return t,l

