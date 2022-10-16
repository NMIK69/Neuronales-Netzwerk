#include "matrix.h"
#include "layer.h"
#include "network.h"
#include "training.h"
#include "benchmarks.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MNIST_DATA_SIZE 60000
#define MNIST_TESTING_SIZE 10000


int main(int argc, char** argv) {
    
    srand(time(NULL));

    int imp = 0;
    int epochs = 10;
    int hidden_layer_num = 1;
    float learning_rate = 0.01;
    int init_per_cmdl = 0;
    int start = 0;
    char* filename = "";
    int sel = -1;
    int epoch_size = 0;
    char* file_mode = "w";


    // how to use commandline: filename, learning rate, epochs, hidden layer number, hidden layer number N size
    if(argc > 1) {
        sel = 0;
        if(strcmp(argv[1], "-t") == 0 && argc > 4) {
            filename = argv[2];
            imp = atoi(argv[3]);
            learning_rate = atof(argv[4]);
            epochs = atoi(argv[5]);


            if(argc > 6 && argc == 7 + atoi(argv[6])) {
                hidden_layer_num = atoi(argv[6]);
                init_per_cmdl = 1;
                start = 7;
            }

        }
        else if(strcmp(argv[1], "-bs") == 0) {
            sel = 1;
            if(argc > 6) {
                
                if(strcmp(argv[2], "-n") == 0) {
                    file_mode = "w";    
                }
                else if(strcmp(argv[2], "-a") == 0) {
                    file_mode = "a";
                } 

                // delete file contetent and make new benchmarks
                filename = argv[3];

                imp = atoi(argv[4]);
                epoch_size = atoi(argv[5]);
                hidden_layer_num = atoi(argv[6]);

                if(argc == hidden_layer_num + 7) {
                    init_per_cmdl = 1;
                    start = 7;
                }
                else {
                    printf("not enought arguments provied... exiting\n");
                    return 0;
                }
            }
        }
        
        else if(strcmp(argv[1], "-bd") == 0) {
            sel = 2;
            if(argc > 4) {
                imp = atoi(argv[2]);
                epoch_size = atoi(argv[3]);
                hidden_layer_num = atoi(argv[4]);


                if(argc == hidden_layer_num + 5) {
                    init_per_cmdl = 1;
                    start = 5;
                }

                else {
                    printf("not enought arguments provied... exiting\n");
                    return 0;
                }
            }
        }

        else if(strcmp(argv[1], "-bfd") == 0) {
            sel = 3;
            if(argc > 4) {
                imp = atoi(argv[2]);
                epoch_size = atoi(argv[3]);
                hidden_layer_num = atoi(argv[4]);


                if(argc == hidden_layer_num + 5) {
                    init_per_cmdl = 1;
                    start = 5;
                }

                else {
                    printf("not enought arguments provied... exiting\n");
                    return 0;
                }
            }
        }

        else if(strcmp(argv[1], "-bfs") == 0) {
            sel = 4;
            if(argc > 6) {
                
                if(strcmp(argv[2], "-n") == 0) {
                    file_mode = "w";    
                }
                else if(strcmp(argv[2], "-a") == 0) {
                    file_mode = "a";
                } 

                filename = argv[3];

                imp = atoi(argv[4]);
                epoch_size = atoi(argv[5]);
                hidden_layer_num = atoi(argv[6]);

                if(argc == hidden_layer_num + 7) {
                    init_per_cmdl = 1;
                    start = 7;
                }
                else {
                    printf("not enought arguments provied... exiting\n");
                    return 0;
                }
            }
        }

    }

    // make default layer in case no arguments have been given
    int* layer_sizes = (int*)malloc((hidden_layer_num + 2) * sizeof(int));
    layer_sizes[0] = 784;
    layer_sizes[1] = 58;
    layer_sizes[hidden_layer_num + 1] = 10;

    if(init_per_cmdl == 1) {
        for(int i = 0; i < hidden_layer_num; i++) {
            layer_sizes[i + 1] = atoi(argv[start + i]);
        }
    }

    if(sel == -1) {
        printf("nothing selected \n");
        return 0;
    }

    else if(sel == 0) { 
        train_and_test_network(imp, filename, learning_rate, epochs, hidden_layer_num + 2, layer_sizes, MNIST_DATA_SIZE, MNIST_TESTING_SIZE);
    }
    else if(sel == 1) {
        benchmark_save(imp, hidden_layer_num + 2, layer_sizes, epoch_size, filename, file_mode);
    }
    else if(sel == 2) {
        benchmark_display(imp, hidden_layer_num + 2, layer_sizes, epoch_size);
    }
    else if(sel == 3) {
        benchmark_fp_only_display(imp, hidden_layer_num + 2, layer_sizes, epoch_size);
    }
    else if(sel == 4) {
        benchmark_fp_only_save(imp, hidden_layer_num + 2, layer_sizes, epoch_size, filename, file_mode);
    }

    free(layer_sizes);

    return 0;
}