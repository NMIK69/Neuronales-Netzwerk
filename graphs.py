from click import option
import matplotlib.pyplot as plt
import numpy as np
import sys

layer_num = 0
N = []
seq = []
para_man = []
para_omp_for = []
seq_simd = []
para_simd = []
filename = ""

def make_compare_graph():

    if(len(sys.argv) == 2):
        filename = sys.argv[1]
    else:
        print("pls provide filename")
        return
    
    f = open(filename, "r")
    
    i = 0
    for line in f:
        temp = line.split(" ")
        layer_num = temp[0]
        #N.insert(0, float(temp[1]))
        N.insert(0, (i+1) * 100)
        seq.insert(0, float(temp[2]))
        para_man.insert(0, float(temp[3]))
        para_omp_for.insert(0, float(temp[4]))
        seq_simd.insert(0, float(temp[5]))
        para_simd.insert(0, float(temp[6]))
        i += 1

    plt.plot(N, seq, label="seq")
    plt.plot(N, para_man, label="para (manual)")
    plt.plot(N, para_omp_for, label="para (automatic)")
    plt.plot(N, seq_simd, label="seq + simd")
    plt.plot(N, para_simd, label="para + simd")

    plt.legend(loc="upper left")
    plt.ylabel("Runtime in seconds")
    plt.xlabel("Size of each layer")
    plt.title("runtime comparison")
    plt.grid()
    plt.show()

    f.close()

def speedup_graph():
    
    pm_su = []
    pomp_su = []
    ssimd_su = []
    psimd_su = []

    for i in range(len(seq)):
        pm_su.append(seq[i]/para_man[i])
        pomp_su.append(seq[i]/para_omp_for[i])
        ssimd_su.append(seq[i]/seq_simd[i])
        psimd_su.append(seq[i]/para_simd[i])

    plt.plot(N, pm_su, label="para (manual)")
    plt.plot(N, pomp_su, label="para (automatic)")
    plt.plot(N, ssimd_su, label="seq + simd")
    plt.plot(N, psimd_su, label="para + simd")

    plt.ylabel("Speedup")
    plt.xlabel("Size of each layer")
    plt.title("speedup comparison")

    plt.legend(loc="center right")
    plt.grid()
    plt.show()


make_compare_graph()
speedup_graph()