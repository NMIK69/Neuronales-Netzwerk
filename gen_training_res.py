import os
import sys

def get_tr():

    lr = sys.argv[1]
    epochs = sys.argv[2]
    
    filename = "training_results_N8"+lr
    main_file = "./nn_main"

    
 
    os.system("make clean")
    os.system("make")

    temp = main_file + " -t " + filename + " 3 " + str(lr) + " " + str(epochs) + " 3 462 462 462"
    os.system(temp)

get_tr()