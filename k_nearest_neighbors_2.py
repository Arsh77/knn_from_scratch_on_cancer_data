import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from math import sqrt
from collections import Counter

def k_nearest(data , predict , k=3):
    if len(data)>=k:
        warnings.warn('K is set to a value less then total voting groups. IDIOTS!!')
    distances=[]
    for group in data :
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            # np.linalg.norm() givs the euclidean distance
            distances.append([euclidean_distance , group])

    votes = [i[1] for i in sorted(distances) [:k]]
    #print(votes)
    votes_result = Counter(votes).most_common(1)[0][0]
    
            
    # k nearest neighbours algorithms
    return votes_result

df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?' , -99999 , inplace=True)
df.drop(['id'] , 1, inplace=True)

# to convert whole data to float and then convert the data to a list
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size=0.2
train_set ={2:[],4:[]}
test_set = {2:[],4:[]}
train_data =full_data[:-int(test_size*len(full_data))] #list is from start to 80 percent of list from the end 
test_data = full_data[-int(test_size*len(full_data)):] # list is 80 percent to 100 percent i.e last 20 percent of data starting from 80% of data onwards
 
for i in train_data:
    train_set[i[-1]].append(i[:-1])
    # yha [i[-1]] represent last feature of data that is class
    # And class have just two values either 2 or 4
    # train_set and test_set are dictionary with feature 2 and 4
    # So the above code will append the values of the 
for i in test_data:  
    test_set[i[-1]].append(i[:-1])

##print(test_set)
##print(20*'#')
##print(train_set)
correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest(train_set , data , k=5)
        if group ==vote:
            correct+=1
        total+=1
print(correct)
print(total)
print('accuracy' , correct/total)        
    
