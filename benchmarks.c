#include <stdio.h>
#include "network.h"
#include "benchmarks.h"
#include <stdlib.h>
#include <omp.h>





void benchmark_display(int imp, int layer_num, int* layer_sizes, int epoch_size) {
    int learning_rate = 1.0;

    /*
        creates a network and displays benchmarks
    */

    // make network
    struct Network* n = make_network(layer_num, layer_sizes, epoch_size, 1);
    read_in_MNIST_img_data(n);
    read_in_MNIST_target(n);

    // display network information
    make_network_information(n);
    display_network_information(n);

    // compare training methods on current network
    compare_training(n, learning_rate, imp);

    delete_network(n);
}


void benchmark_save(int imp, int layer_num, int* layer_sizes, int epoch_size, char* filename, char* file_mode) {
    int learing_rate = 1.0;

    // make network
    struct Network* n = make_network(layer_num, layer_sizes, epoch_size, 1);
    read_in_MNIST_img_data(n);
    read_in_MNIST_target(n);

    // display network information
    make_network_information(n);
    //display_network_information(n);

    compare_and_save(n, learing_rate, filename, imp, file_mode);

    delete_network(n);
}


void benchmark_fp_only_display(int imp, int layer_num, int* layer_sizes, int epoch_size) {
    /*
        creates a network and displays benchmarks
    */

    // make network
    struct Network* n = make_network(layer_num, layer_sizes, epoch_size, 1);
    read_in_MNIST_img_data(n);
    read_in_MNIST_target(n);

    // display network information
    make_network_information(n);
    display_network_information(n);

    // compare training methods on current network
    compare_fp_only(n, imp);

    delete_network(n);
}


void benchmark_fp_only_save(int imp, int layer_num, int* layer_sizes, int epoch_size, char* filename, char* file_mode) {

    // make network
    struct Network* n = make_network(layer_num, layer_sizes, epoch_size, 1);
    read_in_MNIST_img_data(n);
    read_in_MNIST_target(n);

    // display network information
    make_network_information(n);
    //display_network_information(n);

    compare_fp_only_and_save(n, filename, imp, file_mode);

    delete_network(n);
}


void benchmark_autogenerate(int layer_num, int ls, int step, int N, char* filename) {

    int learning_rate = 1.0;
    remove(filename); // remove file to make new benchmarks



    int* layer_sizes = (int*)malloc(layer_num * sizeof(int));
    layer_sizes[0] = 784;
    layer_sizes[layer_num - 1] = 10;

    /*
        The network size gets increased N times. each time every hidden layers gets
        increased by the step amount. and each time the networks performance for the current
        size gets testet and saved to the file. The compare_and_save Function comparse all 
        implementations and saves the to the file.
    */
    for(int i = 0; i < N; i++){

        // increase the size of each hidden layer by the step amount
        for(int j = 1; j < layer_num - 1; j++) {
            layer_sizes[j] = ls;
        }
        ls += step;

        // make network
        struct Network* n = make_network(layer_num, layer_sizes, 1, 1);
        read_in_MNIST_img_data(n);
        read_in_MNIST_target(n);

        make_network_information(n);
        //display_network_information(n);

        // save benchmarks to file
        compare_and_save(n, learning_rate, filename, 0, "w");

        delete_network(n);
    }

    free(layer_sizes);
}