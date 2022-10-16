import matplotlib.pyplot as plt
import sys


def graph_tr():
    suffix = ["3.0", "1.0", "0.01", "0.0001"]
    res = [[] for x in range(len(suffix))]
    filename = "training_results_N7"

    if(len(sys.argv) == 2):
        filename = sys.argv[1]

    epochs = []


    for j in range(len(suffix)):
        tmp_filename = filename+suffix[j]
        f = open(tmp_filename, "r")

        for line in f:
            temp = line.split(" ")
            if j == 0:
                epochs.append(int(temp[0]))
            res[j].append(float(temp[1]))


        f.close()

    for j in range(len(suffix)):
        plt.plot(epochs, res[j], label=suffix[j])

    plt.title("Topologie: 784 116 116 10")
    plt.legend(loc="center right", title="Learningrate")
    plt.ylabel("Vorhersagewahrscheinlichkeit in %")
    plt.xlabel("Epochen")
    plt.grid()
    plt.show()


graph_tr()