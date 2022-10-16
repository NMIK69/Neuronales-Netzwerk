#include <stdio.h>
#include "network.h"
#include "training.h"
#include <omp.h>


void train_and_test_network(int imp, char*filename, float learning_rate, int epochs, int layer_num, int* layer_sizes, int MNIST_DATA_SIZE, int MNIST_TESTING_SIZE) {

    printf("creating network ... \n");
    struct Network* n = make_network(layer_num, layer_sizes, MNIST_DATA_SIZE, MNIST_TESTING_SIZE);

    printf("reading in MNIST img data ... \n");
    read_in_MNIST_img_data(n);

    printf("reading in MNIST lables ... \n");
    read_in_MNIST_target(n);

    printf("Network created \n");

    make_network_information(n);
    display_network_information(n);


    printf("starting training using ");
    double start;
    double end;

    int j = 0;
    //int pg[17] = {1, 2, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 100000};
    int pg[17] = {1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 49, 50, 200, 100000};
    FILE* fp = fopen(filename, "w");

    if(imp == 0) {
        printf("sequential implementation ... \n");
        for(int i = 0; i < epochs; i++) {

            if(i == pg[j]) {
                float perf = get_network_performance(n);
                printf("performance after %d epochs: %.4f%%\n", i, perf);
                j += 1;
            }

            start = omp_get_wtime();
            train_network_seq(n, learning_rate);
            end = omp_get_wtime();
            printf("epoch: %d, time for this epoch: %.4fs ...\n", (i + 1), (end - start));
        }
    }

    else if(imp == 1) {
        printf("para manual implementation ... \n");
        for(int i = 0; i < epochs; i++) {
            start = omp_get_wtime();
            train_network_para(n, learning_rate);
            end = omp_get_wtime();
            printf("epoch: %d, time for this epoch: %.4fs ...\n", (i + 1), (end - start));
        }
    }

    else if(imp == 2) {
        printf("para with omp for implementation ... \n");
        for(int i = 0; i < epochs; i++) {
            start = omp_get_wtime();
            train_network_omp_for(n, learning_rate);
            end = omp_get_wtime();
            printf("epoch: %d, time for this epoch: %.4fs ...\n", (i + 1), (end - start));
        }
    }

    else if(imp == 3) {
        printf("sequential with simd implementation ... \n");
        for(int i = 0; i < epochs; i++) {

            start = omp_get_wtime();
            train_network_simd_seq(n, learning_rate);
            end = omp_get_wtime();
            printf("epoch: %d, time for this epoch: %.4fs ...\n", (i + 1), (end - start));

            if(i+1 == pg[j]) {
                float perf = get_network_performance(n);
                printf("performance after %d epochs: %.4f%%\n", i+1, perf);
                j += 1;
                fprintf(fp, "%d %f\n", i+1, perf);
            }
        }
    }

    else if(imp == 4) {
        printf("para with simd implementation ... \n");
        for(int i = 0; i < epochs; i++) {
            start = omp_get_wtime();
            train_network_smid_para(n, learning_rate);
            end = omp_get_wtime();
            printf("epoch: %d, time for this epoch: %.4fs ...\n", (i + 1), (end - start));
        }
    }

    fclose(fp);
    printf("training done... \n");

    printf("\n starting testing ... \n");

    float performance = get_network_performance(n);

    printf("Network performance: %.2f%% \n", performance);

    delete_network(n);
}