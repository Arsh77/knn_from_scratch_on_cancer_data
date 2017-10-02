# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 03:40:00 2017

@author: ARSHABH SEMWAL
"""

import numpy as np
from sklearn import cross_validation , neighbors
import pandas as pd
import random
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
    confidence =Counter(votes).most_common(1)[0][1]/k
    #print(votes_result , confidence)
    # k nearest neighbours algorithms
    return votes_result , confidence

df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?' , -99999 , inplace=True)
df.drop(['id'] , 1, inplace=True)

# to convert whole data to float and then convert the data to a list
full_data = df.astype(float).values.tolist()
for i in range(20):    
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
            vote , confidence= k_nearest(train_set , data , k=5)
            if group ==vote:
                correct+=1
            #else:
                #print(confidence)
            total+=1
    #print(correct)
    #print(total)
    #print('accuracy' , correct/total)
    accuracy_avg=[]
    accuracy_avg.append(correct/total)

    X=np.array(df.drop(['class'] ,1))
    y=np.array(df['class'])
    X_train , X_test , y_train , y_test=cross_validation.train_test_split(X , y , test_size=0.2)

    clf=neighbors.KNeighborsClassifier()
    clf=clf.fit(X_train ,y_train)
    accuracy_kn=clf.score(X_test , y_test)
    ac_kn=[]
    ac_kn.append(accuracy_kn)

print(sum(accuracy_avg)/len(accuracy_avg))




    
print(sum(ac_kn)/len(ac_kn))


