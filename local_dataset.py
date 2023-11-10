from numpy import load
import numpy as np
import sys

#data = load('data\spi_16.npz')
data = load('g_16.npz')
save_path = "gl_16_2"
recf = 3

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
   

print(len(data['arr_4']))
tmp = data['arr_4']
           
for k in range(2):
    print(k)
    sx = data['arr_5'][k].item()
    sy = data['arr_6'][k].item()
    for i in range(1, 15):
        for j in range(1, 15):
            if not (i <= sx + recf and i >= sx - recf and j <= sy + recf and j >= sy - recf):
                tmp[k][0][i][j] = 0
    print_map(tmp[k])
                


print(len(data['arr_4']))
index = 1000
print_map(data['arr_4'][index])
print_map(data['arr_4'][index + 1])
print_map(data['arr_4'][index + 2])


    
np.savez_compressed(save_path, data['arr_0'], data['arr_1'], data['arr_2'],
                        data['arr_3'], tmp, data['arr_5'], data['arr_6'],
                        data['arr_7'])


#N=(-1, 0), S=(1, 0), E=(0, 1), W=(0, -1), NE=(-1, 1), NW=(-1, -1), SE=(1, 1), SW=(1, -1)


