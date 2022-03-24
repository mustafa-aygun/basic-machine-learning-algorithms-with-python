import matplotlib.pyplot as plt
import numpy as np

from knn import cross_validation, knn

#Loading data to the variables globally to not load it two times.
train_set = np.load('knn/train_set.npy')
train_labels = np.load('knn/train_labels.npy')
test_set = np.load('knn/test_set.npy')
test_labels = np.load('knn/test_labels.npy')

def train(kvalue):
    #Defining empty lists for two different distance types.
    manhattan = []
    euclidean = []
    #For each K value get accuracies.
    for k in range(kvalue):
        manhattan.append(np.around(cross_validation(train_set, train_labels, 10, k+1, 'L1') * 100,9))
        euclidean.append(np.around(cross_validation(train_set, train_labels, 10, k+1, 'L2') * 100,9))
        print(f'K: {k+1}, Manhattan: {manhattan[-1]}, Euclidean: {euclidean[-1]}')
    #Get maximum accuracy K value. 
    maxManhattan = manhattan.index(max(manhattan)) +1
    maxEuclidean = euclidean.index(max(euclidean)) +1
    
    #Plot graphs 
    plotManhattan = plt.figure(1)
    plt.plot(range(1,kvalue+1),manhattan)
    plt.title('Manhattan Accuracy')
    plt.xlabel('K-Value')
    plt.ylabel('Accuracy')
    plt.annotate(f'Max K-value={maxManhattan}',xy=(maxManhattan,manhattan[maxManhattan-1]),xytext=(maxManhattan, manhattan[maxManhattan+1]-10),
            arrowprops=dict(facecolor='black', shrink=0.05),)
    plt.show()
    #Plot graphs 
    plotEuclidean = plt.figure(2)
    plt.plot(range(1,kvalue+1),euclidean)
    plt.title('Euclidean Accuracy')
    plt.xlabel('K-Value')
    plt.ylabel('Accuracy')
    plt.annotate(f'Max K-value={maxEuclidean}',xy=(maxEuclidean,euclidean[maxEuclidean-1]),xytext=(maxEuclidean, euclidean[maxEuclidean+1]-10),
            arrowprops=dict(facecolor='black', shrink=0.05),)
    plt.show()

    #Print best K values.
    print(f'The best K value for Manhattan = {maxManhattan}')
    print(f'The best K value for Euclidean = {maxEuclidean}')

    return maxManhattan, maxEuclidean
    
def test(kManhattan,kEuclidean):
    
    #Do tests with the best Kvalues we found. 
    manhattan = knn(train_set,train_labels,test_set,test_labels,kManhattan,'L1')
    euclidean = knn(train_set,train_labels,test_set,test_labels,kEuclidean,'L2')

    print(f'Test accuracy of Manhattan when k={kManhattan}: {manhattan} ')
    print(f'Test accuracy of Euclidean when k={kEuclidean}: {euclidean} ')

def  main():
    #Main function to call train and test.
    kManhattan, kEuclidean = train(180)
    test(kManhattan,kEuclidean)

if __name__ == "__main__":
    main() #Calling main.
