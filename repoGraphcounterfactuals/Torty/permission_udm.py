
import itertools
import numpy as np


def R_ordinal(X,Y):
    positive_concordant = 0
    negative_concordant = 0
    equal_concordant = 0
    pairs = []
    for i in range(0,len(X)-1):
        j = i
        while j < len(Y)-1:
            j = j + 1
            if i == j:
                continue
            else:
                pairs.append([i,j])
    #print(pairs)
    for index in pairs:
        if X[index[0]] == X[index[1]] and Y[index[0]] == Y[index[1]]:
            equal_concordant += 1
        elif X[index[0]] > X[index[1]] and Y[index[0]] > Y[index[1]]:
            positive_concordant += 1
        elif X[index[0]] < X[index[1]] and Y[index[0]] < Y[index[1]]:
            negative_concordant +=1
                
                
    first = max(positive_concordant,negative_concordant)
    second = min(positive_concordant,negative_concordant)
    standardiser = (len(X)*(len(X)-1)/2)
    return(equal_concordant + (first - second))/(len(pairs))

def R_nominal(X,Y):
    non_concordant = 0
    pairs = []
    for i in range(0,len(X)-1):
        j = i
        while j < len(Y)-1:
            j = j + 1
            if i == j:
                continue
            else:
                pairs.append([i,j])
    for index in pairs:
        if X[index[0]] == X[index[1]] or Y[index[0]] == Y[index[1]]:
            non_concordant += 1
    return non_concordant/(len(pairs))
    
def calculate_entropy(df,X,Y,h,t,v_s):
    N = len(df[X])
    entropy = 0 
    for u in range(0,v_s):
        prob_h = len(df.loc[(df[X]==h) & (df[Y]==u)])/N
        prob_t = len(df.loc[(df[X]==t) & (df[Y]==u)])/N

        prob_th = prob_h + prob_t 
        if prob_th == 0:
            #print("oh dear")
            prob_th = 0.00001
        entropy = entropy + (prob_th * np.log2(prob_th))
    
    entropy = - entropy
    return entropy



def calculate_distance(df,X,Y,h,t,ordinal_indexes,nominal_indexes):
    Y_name = df.columns[Y]
    X_name = df.columns[X]
    v_s = len(df[Y_name].value_counts())
    
    if t == h:
        dist = 0
        
    elif X in ordinal_indexes:
        lower_bound = min(t,h)
        upper_bound = max(t,h)
        entropy = 0 
        for category_index in range(lower_bound, upper_bound):
            entropy = entropy + calculate_entropy(df,X_name,Y_name,category_index, category_index+1,v_s)
        S_Y = -np.log2((1/v_s))
        dist = entropy/S_Y
            
    elif X in nominal_indexes:
        entropy = calculate_entropy(df,X_name,Y_name,h,t,v_s)
        S_Y = -np.log2((1/v_s))
        dist = entropy/S_Y
            
    
    else:
        print("index not recognised")
        return -1
    
    return dist

def distance_algorithm(df, point1, point2, ordinal_indexes, nominal_indexes):
    
    number_of_columns = len(df.columns)
    R_dict = {}
    distance_dict = {}
    for outer_column_index in range(0,number_of_columns):
        h = point1[outer_column_index]
        t = point2[outer_column_index]
        R_dict[outer_column_index] = {}
        temp_distances = []
        for inner_column_index in range(0,number_of_columns):
            if outer_column_index in ordinal_indexes and inner_column_index in ordinal_indexes:
                #ordinal
                R_dict[outer_column_index][inner_column_index] = R_ordinal(df[df.columns[outer_column_index]].values,df[df.columns[inner_column_index]].values)
            else:
                # nominal
                R_dict[outer_column_index][inner_column_index] = R_nominal(df[df.columns[outer_column_index]].values,df[df.columns[inner_column_index]].values)  
            
            temp_distance = calculate_distance(df,outer_column_index, inner_column_index, h, t, ordinal_indexes,nominal_indexes)
            temp_distances.append(temp_distance * R_dict[outer_column_index][inner_column_index])  
        
        distance_dict[outer_column_index] = sum(temp_distances)/len(df.columns)
    
    overall_dist = np.sqrt(np.sum([x**2 for x in distance_dict.values()]))
    return overall_dist


def categorical_distance(df, ordinal_indexes, nominal_indexes):
    distances = []
    permission_dict = {}
    for index1, point1 in df.iterrows():
        permission_dict[index1] = {}
        for index2, point2 in df.iterrows():
            brake = 0
            for col_index in range(0,len(point2)):
                if col_index in ordinal_indexes:
                    if point2[col_index] < point1[col_index]:
                        brake = 1
            if brake == 1:
                permission_dict[index1][index2] = 0
            else:
                permission_dict[index1][index2] = 1

    for index1, point1 in df.iterrows():
        temp_distances = []
        for index2, point2 in df.iterrows():
            if permission_dict[index1][index2] == 1:
            temp_distances.append(distance_algorithm(df, point1, point2, ordinal_indexes,nominal_indexes))
            else:
                temp_distances.append(100)

        distances.append(np.asarray(temp_distances))

    return np.asarray(distances)
    
