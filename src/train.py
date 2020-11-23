import sys, argparse
from TextProcessor import TextProcessor
from NGramLanguageModel import NGramLanguageModel
from PunctuationPredictor import PunctuationPredictor


# print("Python version")
#print (sys.version)
# print("Version info.")
# print (sys.version_info)


def main():
    parser = argparse.ArgumentParser(description='Script to train a language model')
    parser.add_argument("--train", default="../data/ted/TED.en", type=str, help="text file containing the training data")
    parser.add_argument("--valid", default="../data/ted/tst2014.en", type=str, help="text file containing the valid data/ devset")
    parser.add_argument("--test", default="../data/ted/tst2015.en", type=str, help="text file containing the test data")
    parser.add_argument("--model", default="", type=str, help="Language Model")
    parser.add_argument("--text", default="base", type=str, help="Text processor")

    args = parser.parse_args()

    processor = TextProcessor() #instantiates the text preprocessor object
    lm = NGramLanguageModel(3) #Instantiates a n-gram language model


    punc = PunctuationPredictor(lm)

    """ Load training data and train language model"""
    text = getText(args.train)

    """ Process text by the Text Processor"""
    prepro = processor.process(text)

    """ Process text by the Text Processor"""
    print ("Train ", str(lm.getSize()) +  "gram language model with interpolation ....")
    lm.train(prepro)
    print ("Language model trained")

    print ("Validation data")
    valid = getText(args.valid)
    prepro_valid = processor.process(valid)
    lm.getPPL(prepro_valid)
    punc.optThresholds(prepro_valid)


    print ("Test data")
    test = getText(args.test)
    prepro_test = processor.process(test)
    lm.getPPL(prepro_test)
    punc.addPuncuation(prepro_test)


def getText(filename):

    f = open(filename, encoding= "utf-8") #,errors= 'ignore')

    text = [l.strip().split() for l in f.readlines()]
    text = [item for sublist in text for item in sublist]
    #text = [word.lower() for word in text] #no need since this is done in the stemming process of textprocessor
    f.close()
    return text


if __name__ == "__main__":
   main()



