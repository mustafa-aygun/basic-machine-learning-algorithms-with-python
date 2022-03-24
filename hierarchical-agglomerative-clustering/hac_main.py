import numpy as np
from hac import hac
import matplotlib.pyplot as plt 

#Load all datasets.
dataset1 = np.load("hac/dataset1.npy")
dataset2 = np.load("hac/dataset2.npy")
dataset3 = np.load("hac/dataset3.npy")
dataset4 = np.load("hac/dataset4.npy")

#Send given dataset to hac function. The function return clusters.
def train(dataset,criteria,stop):
    A = hac(dataset,criteria,stop)
    return A
#It will take information and clusters to plot them.
def plot(a,criteria,id):
    i = 0
    title = f"Dataset{id+1} / " + criteria
    for cluster in a:
        plt.scatter(cluster[:,0],cluster[:,1],label=i)
        plt.title(title)
        plt.xlabel('X-Axis')
        plt.ylabel('Y-Axis')
        i += 1
    plt.show()  

def  main():
    #Main function to call train and test.
    criteria = ['single_linkage','complete_linkage','average_linkage','centroid_linkage']
    
    datasets = [dataset1,dataset2,dataset3,dataset4]
    stop = 2 #Stop defined two normally.
    for id, dataset in enumerate(datasets):
        A = []
        if(id == 3): #If id is 3 which means dataset4, it will make stop4.
            stop = 4
        for i in range(4): #First will call train for each criteria function and append them to a list.
            A.append(train(dataset, criteria[i],stop))
        for i in range(len(A)): #Then take that list and plot each of them for that dataset.
            plot(A[i],criteria[i],id)

if __name__ == "__main__":
    main() #Calling main.
