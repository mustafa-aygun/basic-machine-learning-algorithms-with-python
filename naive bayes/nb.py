from itertools import count
import numpy as np
import math


def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """

    flat_list = []  #Create empty list
    for sublist in data:
        for item in sublist:
         flat_list.append(item) #Put everything into that empty list
    #Get unique words
    words = np.unique(flat_list)
    word_set = set() #Create a set
    word_set.update(words) #Put words array to that set
    return word_set #Return set

def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    #Get unique labels and their counts
    unique_label, counts = np.unique(train_labels, return_counts=True) 
    pi = {} #Create dict
    total = len(train_labels) #Get total length
    for i in range(len(unique_label)):
        pi[unique_label[i]] = counts[i]/total #Put unique label and probability of it to the dict
    return pi #Return it
    
def estimate_theta(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all the words in vocab and the values are their estimated probabilities given
             the first level class name.
    """
    unique_label = np.unique(train_labels) #Get unique labels
    vocab = vocabulary(train_data) #Create vocab
    train_data = np.array(train_data,dtype=object) #Make it np array
    theta = {} #Create dict
    for label in unique_label:
        theta[label] = {} #Put label 
        train_labels = np.array(train_labels) #Make it np array
        index = np.where(train_labels == label) #Find index of a class
        index = np.ndarray.tolist(index[0]) 
        temp_class_data = np.sum(train_data[index]) #Get examples of that class
        unique_words, word_counts = np.unique(temp_class_data, return_counts=True) #Get unique words and counts
        for j in range(len(vocab)):
            index_word = np.argwhere(unique_words == list(vocab)[j]) #Find position of that vocab
            occurance = 1 #Make occurance 1 at the beginning with directly adding smoothing constant.
            if(index_word != None): #If that word exist in vocab at counts
                occurance += word_counts[index_word[0][0]] 
            theta[label][list(vocab)[j]] = occurance/(np.sum(word_counts)+len(list(vocab))) #Calculate possibility with adding vocab length
    return theta #Return theta
def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """
    scores = [] #Create empty list
    for test in test_data: #For each test example
        list_for_test = [] #Inner list
        for label in theta: #For all class scores
            pi_value = pi[label] #Pi value of class
            score = 0
            score += math.log(pi_value) #Calculate log of it
            for word in test: #For each word in example sentence
                theta_score = 0
                if(word in vocab): #If word in vocab then calculate score
                    theta_score = theta[label][word] 
                    score += math.log(theta_score) #Add score
            temp_tuple = (score, label) #Create a tuple 
            list_for_test.append(temp_tuple) #Add tuple to the list for that class
        scores.append(list_for_test) #Add list to the list for that example
    return scores #Return scores


def main():
    #Read files 
    #Basically just read all the lines and split words according to empty space between them.
    #There is no preprocessing for punctuations or other things.
    file = open("nb_data/train_set.txt", "r", encoding="utf-8")
    train_set = file.read().split("\n")
    temp_train_set = []
    for sentence in train_set:
        temp_train_set.append(sentence.split(" "))
    train_set = temp_train_set

    file = open("nb_data/train_labels.txt", "r", encoding="utf-8")
    train_labels = file.read().split("\n")

    file = open("nb_data/test_set.txt", "r", encoding="utf-8")
    test_set = file.read().split("\n")
    temp_test_set = []
    for sentence in test_set:
        temp_test_set.append(sentence.split(" "))
    test_set = temp_test_set

    file = open("nb_data/test_labels.txt", "r", encoding="utf-8")
    test_labels = file.read().split("\n")

    #Run the functions to get scores
    vocab = vocabulary(train_set)
    pi = estimate_pi(train_labels)
    theta = estimate_theta(train_set,train_labels,vocab)
    scores = test(theta,pi,vocab,test_set)
    count = 0
    
    for i,score in enumerate(scores):
        true_class = max(score) #Get maximum score
        if(true_class[1] == test_labels[i]):
            count += 1 #If it is correct guess increase count by one
    print(f'Accuracy is: {count/len(test_labels)}') #

if __name__ == "__main__":
    main()