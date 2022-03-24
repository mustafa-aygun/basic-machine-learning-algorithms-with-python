import numpy as np

def Euclidean(point1,point2):
    distance = np.sqrt(np.sum(np.square(point1-point2)))
    return distance

def assign_clusters(data, cluster_centers):
    """
    Assigns every data point to its closest (in terms of Euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """
    belongs_clusters = np.ones(len(data))
    for i in range(len(data)):
        min = 100
        for j in range(len(cluster_centers)):
            distance = Euclidean(cluster_centers[j],data[i])
            if(distance < min):
                min = distance
                belongs_clusters[i] = j
    
    return belongs_clusters

def calculate_cluster_centers(data, assignments, cluster_centers, k):
    """
    Calculates cluster_centers such that their squared Euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """
    for i in range(k):
        distance_point = [0,0]
        count = 0
        for j in range(len(data)):
            if(assignments[j] == i ):
                distance_point += data[j]
                count += 1
        if(count != 0 ):
            x = distance_point[0] / count
            y = distance_point[1] / count
            cluster_centers[i] = [x,y]
    
    return cluster_centers



def kmeans(data, initial_cluster_centers):
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """
    cluster_centers = initial_cluster_centers
    previous_objective_function = 0.0
    while(True):
        belongs_clusters = assign_clusters(data,cluster_centers)
        cluster_centers = calculate_cluster_centers(data,belongs_clusters,cluster_centers,len(cluster_centers))
        objective_function = 0.0
        for i in range(len(cluster_centers)):
            for j in range(len(data)):
                if(belongs_clusters[j] == i ):
                    objective_function += np.square(data[j]-cluster_centers[i])
        objective_function = np.sum(objective_function)
        if(objective_function == previous_objective_function):
            return cluster_centers, objective_function/2
        previous_objective_function = objective_function
 
    