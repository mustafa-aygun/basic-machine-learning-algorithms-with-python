import math
import numpy as np
from numpy.core.fromnumeric import sort

def entropy(bucket):
    ent = 0.0
    Sum = sum(bucket)
    if(Sum == 0):
        return 0
    for i in bucket:
        if(i != 0):
            ent -= (i/Sum)*math.log((i/Sum),2)
    return ent
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """


def info_gain(parent_bucket, left_bucket, right_bucket):
    ent_parent = entropy(parent_bucket)
    ent_left = entropy(left_bucket)
    ent_right = entropy(right_bucket)
    sum_parent = sum(parent_bucket)
    sum_left = sum(left_bucket)
    sum_right = sum(right_bucket)
    info_g = ent_parent - ent_left*(sum_left/sum_parent) - ent_right*(sum_right/sum_parent)
    return info_g


    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """


def gini(bucket):
    gini_index = 1.0
    Sum = sum(bucket)
    if(Sum == 0):
        return 0
    for i in bucket:
        gini_index -= math.pow((i/Sum),2)
    return gini_index
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """


def avg_gini_index(left_bucket, right_bucket):
    gini_left = gini(left_bucket)
    gini_right = gini(right_bucket)
    avg_gini = gini_left*(sum(left_bucket)/(sum(left_bucket)+sum(right_bucket)))+gini_right*(sum(right_bucket)/(sum(left_bucket)+sum(right_bucket)))
    return avg_gini
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """


def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    

    indexes = data[:,attr_index].argsort()

    sorted_data = data[indexes]
    sorted_labels = labels[indexes]

    use_data = sorted_data[:,attr_index]
    averages = [None]*(len(use_data)-1)
    for i in range(len(averages)):
        averages[i] = (use_data[i]+use_data[i+1])/2
    

    heuristic_values = []
    if(heuristic_name == 'info_gain'):
        parent_bucket = [0]*num_classes
        for i in labels:
            parent_bucket[i] += 1
        for i in range(len(averages)):
            left_bucket = [0]*num_classes
            right_bucket = [0]*num_classes
            left_split = sorted_labels[:i+1]
            right_split = sorted_labels[i+1:len(sorted_labels)]
            for i in left_split:
                left_bucket[i] += 1
            for i in right_split:
                right_bucket[i] += 1
            info_g = info_gain(parent_bucket,left_bucket,right_bucket)
            heuristic_values.append(info_g)
        L = np.array(list(zip(averages,heuristic_values)))
        return L
    else:
        for i in range(len(averages)):
            left_bucket = [0]*num_classes
            right_bucket = [0]*num_classes
            left_split = sorted_labels[:i+1]
            right_split = sorted_labels[i+1:len(sorted_labels)]
            for i in left_split:
                left_bucket[i] += 1
            for i in right_split:
                right_bucket[i] += 1
            avg_g = avg_gini_index(left_bucket,right_bucket)
            heuristic_values.append(avg_g)
        L = np.array(list(zip(averages,heuristic_values)))
        return L
    """
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """


def chi_squared_test(left_bucket, right_bucket):
    total_values = np.add(left_bucket,right_bucket).tolist()
    expected_values_left = np.sum(left_bucket)/np.sum(total_values)*np.array(total_values)
    expected_values_right = np.sum(right_bucket)/np.sum(total_values)*np.array(total_values)
    left_bucket_difference = np.subtract(left_bucket,expected_values_left).tolist()
    right_bucket_difference = np.subtract(right_bucket,expected_values_right).tolist()
    
    left_bucket_chi = [0]*len(left_bucket)
    right_bucket_chi = [0]*len(right_bucket)

    for i in range(len(left_bucket)):
        if(expected_values_left[i] != 0):
            left_bucket_chi[i] = math.pow(left_bucket_difference[i],2)/expected_values_left[i]
        if(expected_values_right[i] != 0):
            right_bucket_chi[i] = math.pow(right_bucket_difference[i],2)/expected_values_right[i]
            
    chi_value = sum(left_bucket_chi)+sum(right_bucket_chi)
    degree_freedom = (np.count_nonzero(total_values)-1)*1
    return chi_value, degree_freedom
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """
