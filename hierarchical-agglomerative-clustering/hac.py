import numpy as np
def Euclidean(point1,point2):
    distance = np.sqrt(np.sum(np.square(point1-point2)))
    return abs(distance)

def single_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the single linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    distances = []
    for i in range(len(c1)):
        for j in range(len(c2)):
            distances.append(Euclidean(c1[i],c2[j])) 
    return min(distances) #Just calculate distances in that clusters and put an list. Then take minimum value from that list.

def complete_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the complete linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    distances = np.ones((len(c1),len(c2)))
    for i in range(len(c1)):
        for j in range(len(c2)):
            distances[i][j] = Euclidean(c1[i],c2[j])
    return np.max(distances) #Just calculate distances in that clusters and put an list. Then take maximum value from that list.


def average_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the average linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    distances = np.ones((len(c1),len(c2)))
    for i in range(len(c1)):
        for j in range(len(c2)):
            distances[i][j] = Euclidean(c1[i],c2[j])
    return np.mean(distances) #Just calculate distances in that clusters and put an list. Then take man of these values.


def centroid_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the centroid linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    #First calculate centroid of each cluster. Basically collecting all points and divide it to total length.
    c1_total = 0
    for i in range(len(c1)):
        c1_total += c1[i]
    c1_total[0] = c1_total[0] / len(c1)
    c1_total[1] = c1_total[1] / len(c1)

    c2_total = 0
    for i in range(len(c2)):
        c2_total += c2[i]
    c2_total[0] = c2_total[0] / len(c2)
    c2_total[1] = c2_total[1] / len(c2)
    #Then take distance and return it.
    distance = Euclidean(c1_total,c2_total)
    return distance


def hac(data, criterion, stop_length):
    """
    Applies hierarchical agglomerative clustering algorithm with the given criterion on the data
    until the number of clusters reaches the stop_length.
    :param data: An (N, D) shaped numpy array containing all of the data points.
    :param criterion: A function. It can be single_linkage, complete_linkage, average_linkage, or
    centroid_linkage
    :param stop_length: An integer. The length at which the algorithm stops.
    :return: A list of numpy arrays with length stop_length. Each item in the list is a cluster
    and a (Ni, D) sized numpy array.
    """
    #Deciding function
    if(criterion == 'single_linkage'):
        distance_function = single_linkage
    elif(criterion == 'complete_linkage'):
        distance_function = complete_linkage
    elif(criterion == 'average_linkage'):
        distance_function = average_linkage
    else:
        distance_function = centroid_linkage 

    #Creating an empty list and put all points in it as array list.
    A = []
    for point in data:
        A.append(np.array([point]))
    
    while(len(A) > stop_length): #Continue till stop_length.
        min_distance = float('inf') #Initialize a min distance.
        min_cluster_1 = None #Initialise two cluster 
        min_cluster_2 = None
        index_1 = 0
        min_index_1 = -1 #Initialise two variable to keep track of index of min distance.
        min_index_2 = -1
        for cluster_1 in A:
            index_2 = 0
            for cluster_2 in A:
                if(index_1 != index_2): #If distance result won't be zero, it means if they are not same clusters we are continue.      
                    dist = distance_function(cluster_1,cluster_2)
                    if(dist < min_distance): #If calculated distance is lower than min distance.
                        min_distance = dist #Make it new min distance.
                        min_cluster_1 = cluster_1 #Give current clusters as min clusters
                        min_cluster_2 = cluster_2
                        min_index_1 = index_1 #Give current indexs as min clusters index.
                        min_index_2 = index_2
                index_2 += 1
            index_1 += 1 
        if(min_index_2 > min_index_1): #If min_index_2 > min_index_1 first pop min_index_2 to avoid changing of index number of min_index_1
            A.pop(min_index_2) #Pop two min index to put them together. We are basically deleting them to put again.
            A.pop(min_index_1)
        else:
            A.pop(min_index_1) #If min_index_2 < min_index_1 first pop min_index_1 to avoid changing of index number of min_index_2
            A.pop(min_index_2)
        A.append(np.concatenate((min_cluster_1,min_cluster_2))) #Then append these two cluster together after poping them out.
    return A
        

