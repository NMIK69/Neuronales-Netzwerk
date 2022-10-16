from click import option
import os
import sys

def make_and_get_performance():
    c2 = ["-O3"]
    c3 = sys.argv[1::]
    
    filename = "benchmarks_file"
    main_file = "./nn_main"
    options = ["-t", "-bs", "-n", "-a", "-bd", "-bfs", "-bfd"]
    layers = 3
    end = 1000

    for option in c2:
        temp = option.split()
        co_filename = filename
        for x in temp:
            co_filename = co_filename+x

        os.system("make clean")
        os.system("make OPTIMIZE="+'"'+option+'"')

        for i in range(100, end, 100):
            if(i == 100):
                temp = main_file + " " + options[1] + " " + options[2] + " " + co_filename + " -1 1000 " + str(layers)
            else:
                temp = main_file + " " + options[1] + " " + options[3] + " " + co_filename + " -1 1000 " + str(layers)
            for j in range(layers):
                temp += " " + str(i)

            os.system(temp)

make_and_get_performance()