import numpy as np


def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """
    trellis_diagram = np.zeros((len(pi),len(O))) #Create diagram matrix
    #It is a nested for loop which is N^2*T
    for i in range(len(O)): #Sequence length
        for j in range(len(pi)): #State length
            if(i == 0): #If i equals to 0 that means we are at the first step and just calculate initial and observation probabilities
                trellis_diagram[j,i] = pi[j]*B[j,O[i]]
            else: #Else calculate all previous step possibilities and add them.
                for k  in range(len(pi)): #Multiply observastion, state transition and previous step probabilties
                    trellis_diagram[j,i] += B[j,O[i]]*A[k,j]*trellis_diagram[k,i-1]
    probability = trellis_diagram.sum(axis=0) #Sum of all columns
    return probability[len(O)-1], trellis_diagram #Return diagram and the last column


def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
    #Initialising empty arrays
    state_sequence = np.zeros(len(O))
    trellis_diagram = np.zeros((len(pi),len(O)))
    #It is a nested for loop which is N^2*T
    for i in range(len(O)): #Sequence length
        for j in range(len(pi)): #State length
            max = 0
            if(i == 0): #If i equals to 0 that means we are at the first step and just calculate initial and observation probabilities
                trellis_diagram[j,i] = pi[j]*B[j,O[i]]
            else:  #Else calculate all previous step possibilities get the max one.
                for k  in range(len(pi)): #State length
                    temp = B[j,O[i]]*A[k,j]*trellis_diagram[k,i-1] #Calculate probability
                    if(temp > max): #Check if it is bigger than max
                        max = temp 
                        trellis_diagram[j,i] = max #Assign it to diagram if it is bigger 

    state_sequence = np.argmax(trellis_diagram,axis=0) #Get maximum index at each column
    return state_sequence, trellis_diagram #Return diagram and sequence.