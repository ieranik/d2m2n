from numpy import load
import numpy as np
import sys

def print_map(a):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                if a[i][j][k] == 0:
                    print(". ", end = '')
                else:  
                    print("0 ", end = '')
            print("")
        print("")
   


data = load('g_20.npz', allow_pickle=True)
print(len(data['arr_0']))
print(len(data['arr_4']))
var = input()
index = 100
print_map(data['arr_0'][index])
print(data['arr_1'][index], data['arr_2'][index])
print(data['arr_3'][index])


#N=(-1, 0), S=(1, 0), E=(0, 1), W=(0, -1), NE=(-1, 1), NW=(-1, -1), SE=(1, 1), SW=(1, -1)

#print(data['arr_0'][0].shape)
#for i in range(100):
#    print(data['arr_0'][i])

for line in sys.stdin:
    print(line)
#for item in lst:
    #print(item)
    #print(data[item][2])
