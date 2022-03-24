import numpy as np

#Distance calculation of Manhattan
def Manhattan(train_data_point,test_instance_point):
    distance = test_instance_point-train_data_point
    sum = np.sum(abs(distance)) #Getting absolute value of it to get rid of negative values. 
    return sum
#Distance calculation of Euclidean
def Euclidean(train_data_point,test_instance_point):
    distance = np.sqrt(np.sum(np.square(test_instance_point-train_data_point)))
    return distance

def calculate_distances(train_data, test_instance, distance_metric):
    """
    Calculates Manhattan (L1) / Euclidean (L2) distances between test_instance and every train instance.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data.
    :param test_instance: A (D, ) shaped numpy array.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: An (N, ) shaped numpy array that contains distances.
    """
    #If distance_metric is L1, then it is Manhattan function. Otherwise, it is Euclidean function.
    if(distance_metric == 'L1'):
        distance_function = Manhattan
    else:
        distance_function = Euclidean

    #Create a numpy array with random ones. We will fill it later anyways.
    distance_array = np.ones(len(train_data)) 
    for i in range(len(train_data)): #Calculate the distances one by one and return the array.
        distance_array[i] = distance_function(train_data[i],test_instance)
    return distance_array



def majority_voting(distances, labels, k):
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """
    
    #I just wanted to be sure arrays will be numpy array. 
    distances = np.array(distances)
    labels = np.array(labels) 
    #Getting indexes according to distance sort.
    indexs = distances.argsort()
    #Changing positions of labels array according to sort. It will be change order in same way.
    labels = labels[indexs]
    #Get first k element.
    temp_labels = labels[:k]
    #Getting most repeated one from the list and return it.
    max_label = np.bincount(temp_labels).argmax()
    return max_label


def knn(train_data, train_labels, test_data, test_labels, k, distance_metric):
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. The calculated accuracy.
    """
    #Initialise number.
    correct_guess = 0
    for i in range(len(test_data)): 
        #Calculate distances with new value and get label with majority_voting.
        distances = calculate_distances(train_data,test_data[i],distance_metric)
        label = majority_voting(distances,train_labels,k)
        #If estimated label is correct increase correct_guess.
        if (label == test_labels[i]):
            correct_guess += 1
    #Calculate accuracy.
    accuracy = correct_guess / (i+1)
    return accuracy

def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """

    #Split data according to k_fold.
    train_data_groups = np.array_split(whole_train_data, k_fold)
    train_labels_groups = np.array_split(whole_train_labels, k_fold)
  
    #Move validation index out and concatenate before and after as one array.
    train_data = np.concatenate(train_data_groups[:validation_index:]+train_data_groups[validation_index+1::])
    train_labels = np.concatenate(train_labels_groups[:validation_index:]+train_labels_groups[validation_index+1::])

    #Get validation data and return all of them
    validation_data = train_data_groups[validation_index]
    validation_labels = train_labels_groups[validation_index]
    return train_data, train_labels, validation_data, validation_labels


def cross_validation(whole_train_data, whole_train_labels, k_fold, k, distance_metric):
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k_fold: An integer.
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. Average accuracy calculated.
    """
    #Initialise variables.
    average_accuracy = 0
    accuracy = 0
    for i in range(k_fold):
        #First split data then call knn function to calculate accuracy.
        train_data, train_labels, validation_data, validation_labels = split_train_and_validation(whole_train_data, whole_train_labels, i, k_fold)
        temp_accuracy = knn(train_data,train_labels,validation_data,validation_labels,k,distance_metric)
        #Add accuracy to total.
        accuracy += temp_accuracy
    #Calculate average_accuracy and return it.
    average_accuracy = accuracy / k_fold
    return average_accuracy