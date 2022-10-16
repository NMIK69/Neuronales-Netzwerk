import matplotlib.pyplot as plt
import sys

layer_num = 0
N = []
seq = []
para_man = []
para_omp_for = []
seq_simd = []
para_simd = []
filename = ""


spm = []
spa = []
sps = []
ss = []
sss = []

def make_compare_graph_dco():

    co = ["-O0", "-O2","-O3-ffast-math"]
    titles = ["seq", "para (manual)", "para (automatic)", "seq + simd", "para + simd"]
    co_len = len(co)
    filename = "benchmarks_file"

    seq = [[] for x in range(co_len)]
    para_man = [[] for x in range(co_len)]
    para_omp_for = [[] for x in range(co_len)]
    seq_simd = [[] for x in range(co_len)]
    para_simd = [[] for x in range(co_len)]

    spm = [[] for x in range(co_len)]
    spa = [[] for x in range(co_len)]
    sps = [[] for x in range(co_len)]
    sss = [[] for x in range(co_len)]
    ss = [[] for x in range(co_len)]

    for j in range(co_len):
        co_filename = filename+co[j]
        print(co_filename)
        f = open(co_filename, "r")

        i = 0
        for line in f:
            temp = line.split(" ")
            layer_num = temp[0]
            #N.insert(0, float(temp[1]))
            if(j == 0):
                N.insert(0, (i+1) * 1000)

            seq[j].insert(0, float(temp[2]))
            para_man[j].insert(0, float(temp[3]))
            para_omp_for[j].insert(0, float(temp[4]))
            seq_simd[j].insert(0, float(temp[5]))
            para_simd[j].insert(0, float(temp[6]))

            spm[j].append(0)
            spa[j].append(0)
            sps[j].append(0)
            sss[j].append(0)
            ss[j].append(0)

            i += 1

        f.close()

    for i in range(1, len(seq)):
        for j in range(len(seq[i])):
            spm[i][j] = para_man[0][j]/para_man[i][j]
            spa[i][j] = para_omp_for[0][j]/para_omp_for[i][j]
            sps[i][j] = para_simd[0][j]/para_simd[i][j]
            sss[i][j] = seq_simd[0][j]/seq_simd[i][j]
            ss[i][j] = seq[0][j]/seq[i][j]

    #for j in range(5):
    #    for i in range(co_len):
    #        if j == 0: plt.plot(N, seq[i], label=co[i])
    #        if j == 1: plt.plot(N, para_man[i], label=co[i])
    #        if j == 2: plt.plot(N, para_omp_for[i], label=co[i])
    #        if j == 3: plt.plot(N, seq_simd[i], label=co[i])
    #        if j == 4: plt.plot(N, para_simd[i], label=co[i])
    #    
    #    plt.title(titles[j])
    #    plt.legend(loc="upper left")
    #    plt.ylabel("Runtime in seconds")
    #    plt.xlabel("Size of each layer")
#
    #    plt.grid()
    #    plt.show()

    for j in range(5):
        for i in range(1, co_len):
            if j == 0: plt.plot(N, ss[i], label=co[i])
            if j == 1: plt.plot(N, spm[i], label=co[i])
            if j == 2: plt.plot(N, spa[i], label=co[i])
            if j == 3: plt.plot(N, sss[i], label=co[i])
            if j == 4: plt.plot(N, sps[i], label=co[i])
        
        plt.title(titles[j])
        plt.legend(loc="upper right")
        plt.ylabel("speedup")
        plt.xlabel("Size of each layer")

        plt.grid()
        plt.show()



make_compare_graph_dco()


