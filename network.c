#include "network.h"
#include "matrix.h"
#include "layer.h"
#include "network_learn_para.h"
#include "network_learn_omp_for.h"
#include "network_learn_seq.h"
#include "network_learn_simd_seq.h"
#include "network_learn_simd_para.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>


#define img_width 28
#define img_height 28



struct Network* make_network(int layer_num, int* layer_sizes, int training_data_size, int testing_data_size) {
    
    // make network
    struct Network* nw = (struct Network*)malloc(sizeof(struct Network));
    nw->layer_num = layer_num;
    nw->layer_sizes = layer_sizes;
    nw->training_data_size = training_data_size;
    nw->testing_data_size = testing_data_size;
    nw->testing_index = 0;
    nw->training_index = 0;
    nw->computation_index = 0;
    nw->number_of_nodes = 0;
    nw->number_of_weights = 0;
    nw->fp_itterations = 0;
    nw->bp_itterations = 0;


    //make layers
    // input output hidden layers
    nw->layers = (struct Layer**)malloc(layer_num * sizeof(struct Layer*));
    for(int i = 0; i < layer_num; i++) {
        nw->layers[i] = make_layer(layer_sizes[i]);
    }
    nw->l1 = nw->layers[0];

    // training img data
    nw->MNIST_img_data = (struct Layer**)malloc(training_data_size * sizeof(struct Layer*));
    for(int i = 0; i < training_data_size; i++) {
        nw->MNIST_img_data[i] = make_layer(layer_sizes[0]);
    }

    // training lable_data
    nw->MNIST_target = (struct Layer**)malloc(training_data_size * sizeof(struct Layer*));
    for(int i = 0; i < training_data_size; i++) {
        nw->MNIST_target[i] = make_layer(layer_sizes[layer_num - 1]);
    }

    // testing img data
    nw->Mnist_testing_img_date = (struct Layer**)malloc(testing_data_size * sizeof(struct Layer*));
    for(int i = 0; i < testing_data_size; i++) {
        nw->Mnist_testing_img_date[i] = make_layer(layer_sizes[0]);
    }


    // testing lable data
    nw->MNIST_testing_target = (struct Layer**)malloc(testing_data_size * sizeof(struct Layer*));
    for(int i = 0; i < testing_data_size; i++) {
        nw->MNIST_testing_target[i] = make_layer(layer_sizes[layer_num - 1]);
    }


    //make matricies
    nw->weights = (struct Matrix**)malloc( (layer_num - 1) * sizeof(struct Matrix*));
    nw->delta_weights = (struct Matrix**)malloc( (layer_num - 1) * sizeof(struct Matrix*));
    for(int i = 0; i < layer_num - 1; i++) {
        nw->weights[i] = make_matrix(layer_sizes[i + 1], layer_sizes[i]);
        nw->delta_weights[i] = make_matrix(layer_sizes[i + 1], layer_sizes[i]);
        randomize_matrix(nw->weights[i], 1);
    }

    // make target
    nw->target = make_layer(layer_sizes[layer_num - 1]);


    // this is used in backpropagation and stores computations in the previous layer that can be reused
    nw->computations = (struct Layer**)malloc((layer_num - 1) * sizeof(struct Layer*));
    for(int i = 0; i < layer_num - 1; i++) {
        nw->computations[i] = make_layer(layer_sizes[layer_num - 1 - i]);
    }


    return nw;
}


void delete_network(struct Network* nw) {
    
    for(int i = 1; i < nw->layer_num; i++) {
        delete_layer(nw->layers[i]);
    }
    free(nw->layers);
    free(nw->l1);
   

    // training img data
    for(int i = 0; i < nw->training_data_size; i++) {
        delete_layer(nw->MNIST_img_data[i]);
    }
    free(nw->MNIST_img_data);
//

    // training lable_data
    for(int i = 0; i < nw->training_data_size; i++) {
        delete_layer(nw->MNIST_target[i]);
    }
    free(nw->MNIST_target);
//

    // testing img data
    for(int i = 0; i < nw->testing_data_size; i++) {
        delete_layer(nw->Mnist_testing_img_date[i]);
    }
    free(nw->Mnist_testing_img_date);

//
    // testing lable data
    for(int i = 0; i < nw->testing_data_size; i++) {
        delete_layer(nw->MNIST_testing_target[i]);
    }
    free(nw->MNIST_testing_target);


    //make matricies
    for(int i = 0; i < nw->layer_num - 1; i++) {
        delete_matrix(nw->weights[i]);
        delete_matrix(nw->delta_weights[i]);
    }
    free(nw->weights);
    free(nw->delta_weights);
    
    // make target
    //delete_layer(nw->target);

    // this is used in backpropagation and stores computations in the previous layer that can be reused
    for(int i = 0; i < nw->layer_num - 1; i++) {
        delete_layer(nw->computations[i]);
    }
    free(nw->computations);


    free(nw);
}


void make_network_information(struct Network* nw) {
    nw->bp_itterations = nw->layer_sizes[nw->layer_num - 2] * nw->layer_sizes[nw->layer_num - 1];

    for(int i = 0; i < nw->layer_num; i++) {
        nw->number_of_nodes += nw->layer_sizes[i];

        if(i > 0) {
            nw->number_of_weights += nw->layer_sizes[i - 1];
            nw->fp_itterations += nw->layer_sizes[i - 1] * nw->layer_sizes[i];
        }

        if(i > 0 && i < nw->layer_num - 1) {
            nw->bp_itterations += nw->layer_sizes[i] * (nw->layer_sizes[i - 1] + nw->layer_sizes[i + 1]);
        }
    }
}


void display_network_information(struct Network* nw) {

    printf("*** Network Information *** \n\n");
    // print layers
    for(int i = 0; i < nw->layer_num; i++) {
        if(i == nw->layer_num -1) {
            printf("%d\n\n", nw->layer_sizes[i]);
        }
        else {
            printf("%d ---- ", nw->layer_sizes[i]);
        }
    }

    // print number of nodes
    printf("total nodes: %lld\n\n", nw->number_of_nodes);

    // print number of weights
    printf("total weights: %lld\n\n", nw->number_of_weights);

    // print number of itterations for forward prop (one pass not total)
    printf("forward prop itterations: %lld\n\n", nw->fp_itterations);

    // print number of itterations for back prop (one pass not total)
    printf("backprop itterations: %lld\n\n", nw->bp_itterations);
    // print number of itterations for update weights (one pass not total)
    printf("update weights ittreations: %lld\n\n", nw->number_of_weights);

    // print number of itterations for one pass
    printf("itterations for one pass: %lld \n\n", (nw->number_of_weights + nw->fp_itterations + nw->bp_itterations));
}


int read_in_MNIST_img_data(struct Network* nw) {
    int bytes_read;
    char* file_name = "train-images.idx3-ubyte";
    int fid = open(file_name, O_RDONLY);
    int x = 0;
    
    lseek(fid, 16, SEEK_CUR);
    
    for(int i = 0; i < nw->training_data_size; i++) {
        for(int j = 0; j < nw->layer_sizes[0]; j++) {
            bytes_read = read(fid, &x, 1);
            nw->MNIST_img_data[i]->a[j] = ( (float)x / 255.0);
        }
    }
    close(fid);
    

    file_name = "t10k-images.idx3-ubyte";
    fid = open(file_name, O_RDONLY);
    x = 0;

    lseek(fid, 16, SEEK_CUR);

    for(int i = 0; i < nw->testing_data_size; i++) {
        for(int j = 0; j < nw->layer_sizes[0]; j++) {
            bytes_read = read(fid, &x, 1);
            nw->Mnist_testing_img_date[i]->a[j] = ( (float)x / 255.0);
        }
    }
    close(fid);

    return 0;
}


int read_in_MNIST_target(struct Network* nw) {
    int bytes_read;
    char* file_name = "train-labels.idx1-ubyte";
    int fid = open(file_name, O_RDONLY);
    int x = 0;
    
    lseek(fid, 8, SEEK_CUR);
    
    for(int i = 0; i < nw->training_data_size; i++) {
        bytes_read = read(fid, &x, 1);
        for(int j = 0; j < nw->layer_sizes[nw->layer_num - 1]; j++) {
            nw->MNIST_target[i]->a[j] = 0;   
        }
        nw->MNIST_target[i]->a[x] = 1.0;
    }
    close(fid);


    file_name = "t10k-labels.idx1-ubyte";
    fid = open(file_name, O_RDONLY);
    x = 0;
    
    lseek(fid, 8, SEEK_CUR);
    
    for(int i = 0; i < nw->testing_data_size; i++) {
        bytes_read = read(fid, &x, 1);
        for(int j = 0; j < nw->layer_sizes[nw->layer_num - 1]; j++) {
            nw->MNIST_testing_target[i]->a[j] = 0;
        }
        nw->MNIST_testing_target[i]->a[x] = 1.0;
    }
    close(fid);


    return 0;
}

float get_network_performance(struct Network* nw) {
    
    int prediction;
    int target;

    int correct = 0;

    for(int i = 0; i < nw->testing_data_size; i++) {

        nw->layers[0] = nw->Mnist_testing_img_date[i];
        nw->target = nw->MNIST_testing_target[i];


        forward_prop_para(nw); 

        prediction = get_prediction(nw->layers[nw->layer_num - 1]);
        target = get_target(nw->target);

        if(target == prediction) {
            correct += 1;
        }
    }

    float performance = ( (float)correct / (float)nw->testing_data_size ) * 100.0;
    return performance;
}


void train_network_para(struct Network* nw, float learning_rate) {

    for(int i = 0; i < nw->training_data_size; i++) {
        nw->layers[0] = nw->MNIST_img_data[i];
        nw->target = nw->MNIST_target[i];


        forward_prop_para(nw);
        backwards_prop_para(nw);
        update_weights_para(nw, &learning_rate); 
    }

}


void train_network_omp_for(struct Network* nw, float learning_rate) {

    for(int i = 0; i < nw->training_data_size; i++) {
        nw->layers[0] = nw->MNIST_img_data[i];
        nw->target = nw->MNIST_target[i];


        forward_prop_omp_for(nw);
        backwards_prop_omp_for(nw);
        update_weights_omp_for(nw, &learning_rate); 
    }
}

void train_network_seq(struct Network* nw, float learning_rate) {

    for(int i = 0; i < nw->training_data_size; i++) {
        nw->layers[0] = nw->MNIST_img_data[i];
        nw->target = nw->MNIST_target[i];


        forward_prop_seq(nw);
        backwards_prop_seq(nw);
        update_weights_seq(nw, &learning_rate); 
    }
}


void train_network_simd_seq(struct Network* nw, float learning_rate) {

    for(int i = 0; i < nw->training_data_size; i++) {
        nw->layers[0] = nw->MNIST_img_data[i];
        nw->target = nw->MNIST_target[i];


        forward_prop_simd_seq(nw);
        backwards_prop_simd_seq(nw);
        update_weights_simd_seq(nw, &learning_rate);
    }
}


void train_network_smid_para(struct Network* nw, float learning_rate) {
    for(int i = 0; i < nw->training_data_size; i++) {
        nw->layers[0] = nw->MNIST_img_data[i];
        nw->target = nw->MNIST_target[i];


        forward_prop_simd_para(nw);
        backwards_prop_simd_para(nw);
        update_weights_simd_para(nw, &learning_rate); 
    }
}


void compare_training(struct Network* nw, float learning_rate, int imp) {

    double start;
    double end;
    int epoch_size = nw->training_data_size;

    // sequential time
    if(imp == -1 || imp == 0) {
        start = omp_get_wtime();
        train_network_seq(nw, learning_rate);
        end = omp_get_wtime();
        printf("seq: %.4fs\n", ((end - start)/ epoch_size));
    }
    

    // para manual mode time
    if(imp == -1 || imp == 1) {
        start = omp_get_wtime();
        train_network_para(nw, learning_rate);
        end = omp_get_wtime();
        printf("para man: %.4fs\n", ((end - start)/ epoch_size));
    }
        

    // para with omp for time
    if(imp == -1 || imp == 2) {
        start = omp_get_wtime();
        train_network_omp_for(nw, learning_rate);
        end = omp_get_wtime();
        printf("omp for: %.4fs\n", ((end - start)/ epoch_size));
    }

    // seq simd time
    if(imp == -1 || imp == 3) {
        start = omp_get_wtime();
        train_network_simd_seq(nw, learning_rate);
        end = omp_get_wtime();
        printf("seq simd: %.4fs \n", ((end - start)/ epoch_size));
    }

    // para with simd time
    if(imp == -1 || imp == 4) {
        start = omp_get_wtime();
        train_network_smid_para(nw, learning_rate);
        end = omp_get_wtime();
        printf("para simd: %.4fs \n", ((end - start)/ epoch_size));
    }

}


void compare_and_save(struct Network* nw, float learning_rate, char* filename, int imp, char* file_mode){

    double start;
    double end;
    int epoch_size = nw->training_data_size;

    int x = nw->bp_itterations + nw->fp_itterations + nw->number_of_weights;
    

    FILE* fp = fopen(filename, file_mode);
    fprintf(fp, "%d %d", nw->layer_num - 2, x);

    // sequential time
    if(imp == -1 || imp == 0) {
        start = omp_get_wtime();
        train_network_seq(nw, learning_rate);
        end = omp_get_wtime();
        fprintf(fp, " %.4f", ((end - start)/ epoch_size));
    }

    // para manual mode time
    if(imp == -1 || imp == 1) {
        start = omp_get_wtime();
        train_network_para(nw, learning_rate);
        end = omp_get_wtime();
        fprintf(fp, " %.4f", ((end - start)/ epoch_size));
    }

    // para with omp for time
    if(imp == -1 || imp == 2) {
        start = omp_get_wtime();
        train_network_omp_for(nw, learning_rate);
        end = omp_get_wtime();
        fprintf(fp, " %.4f", ((end - start)/ epoch_size));
    }

    // seq simd time
    if(imp == -1 || imp == 3) {
        start = omp_get_wtime();
        train_network_simd_seq(nw, learning_rate);
        end = omp_get_wtime();
        fprintf(fp, " %.4f", ((end - start)/ epoch_size));
    }

    // para with simd time
    if(imp == -1 || imp == 4) {
        start = omp_get_wtime();
        train_network_smid_para(nw, learning_rate);
        end = omp_get_wtime();
        fprintf(fp, " %.4f", ((end - start)/ epoch_size));
    }

    fprintf(fp, "\n");
    fclose(fp);
}


void compare_fp_only(struct Network* nw, int imp) {

    double start;
    double end;
    int epoch_size = nw->training_data_size;

    // sequential time
    if(imp == -1 || imp == 0) {
        start = omp_get_wtime();

        for(int i = 0; i < nw->training_data_size; i++) {
            nw->layers[0] = nw->MNIST_img_data[i];
            nw->target = nw->MNIST_target[i];

            forward_prop_seq(nw);
        }

        end = omp_get_wtime();
        printf("seq: %.4fs\n", ((end - start)/ epoch_size));
    }
    

    // para manual mode time
    if(imp == -1 || imp == 1) {
        start = omp_get_wtime();
        for(int i = 0; i < nw->training_data_size; i++) {
            nw->layers[0] = nw->MNIST_img_data[i];
            nw->target = nw->MNIST_target[i];

            forward_prop_para(nw);
        }
        end = omp_get_wtime();
        printf("para man: %.4fs\n", ((end - start)/ epoch_size));
    }
        

    // para with omp for time
    if(imp == -1 || imp == 2) {
        start = omp_get_wtime();
        for(int i = 0; i < nw->training_data_size; i++) {
            nw->layers[0] = nw->MNIST_img_data[i];
            nw->target = nw->MNIST_target[i];

            forward_prop_omp_for(nw);
        }
        end = omp_get_wtime();
        printf("omp for: %.4fs\n", ((end - start)/ epoch_size));
    }

    // seq simd time
    if(imp == -1 || imp == 3) {
        start = omp_get_wtime();
        for(int i = 0; i < nw->training_data_size; i++) {
            nw->layers[0] = nw->MNIST_img_data[i];
            nw->target = nw->MNIST_target[i];

            forward_prop_simd_seq(nw);
        }
        end = omp_get_wtime();
        printf("seq simd: %.4fs \n", ((end - start)/ epoch_size));
    }

    // para with simd time
    if(imp == -1 || imp == 4) {
        start = omp_get_wtime();
        for(int i = 0; i < nw->training_data_size; i++) {
            nw->layers[0] = nw->MNIST_img_data[i];
            nw->target = nw->MNIST_target[i];

            forward_prop_simd_para(nw);
        }
        end = omp_get_wtime();
        printf("para simd: %.4fs \n", ((end - start)/ epoch_size));
    }
}


void compare_fp_only_and_save(struct Network* nw, char* filename, int imp, char* file_mode) {
    double start;
    double end;
    int epoch_size = nw->training_data_size;

    int x = nw->bp_itterations + nw->fp_itterations + nw->number_of_weights;
    

    FILE* fp = fopen(filename, file_mode);
    fprintf(fp, "%d %d", nw->layer_num - 2, x);

    // sequential time
    if(imp == -1 || imp == 0) {
        start = omp_get_wtime();
        for(int i = 0; i < nw->training_data_size; i++) {
            nw->layers[0] = nw->MNIST_img_data[i];
            nw->target = nw->MNIST_target[i];

            forward_prop_seq(nw);
        }
        end = omp_get_wtime();
        fprintf(fp, " %.4f", ((end - start)/ epoch_size));
    }

    // para manual mode time
    if(imp == -1 || imp == 1) {
        start = omp_get_wtime();
        for(int i = 0; i < nw->training_data_size; i++) {
            nw->layers[0] = nw->MNIST_img_data[i];
            nw->target = nw->MNIST_target[i];

            forward_prop_para(nw);
        }
        end = omp_get_wtime();
        fprintf(fp, " %.4f", ((end - start)/ epoch_size));
    }

    // para with omp for time
    if(imp == -1 || imp == 2) {
        start = omp_get_wtime();
        for(int i = 0; i < nw->training_data_size; i++) {
            nw->layers[0] = nw->MNIST_img_data[i];
            nw->target = nw->MNIST_target[i];

            forward_prop_omp_for(nw);
        }
        end = omp_get_wtime();
        fprintf(fp, " %.4f", ((end - start)/ epoch_size));
    }

    // seq simd time
    if(imp == -1 || imp == 3) {
        start = omp_get_wtime();
        for(int i = 0; i < nw->training_data_size; i++) {
            nw->layers[0] = nw->MNIST_img_data[i];
            nw->target = nw->MNIST_target[i];

            forward_prop_simd_seq(nw);
        }
        end = omp_get_wtime();
        fprintf(fp, " %.4f", ((end - start)/ epoch_size));
    }

    // para with simd time
    if(imp == -1 || imp == 4) {
        start = omp_get_wtime();
        for(int i = 0; i < nw->training_data_size; i++) {
            nw->layers[0] = nw->MNIST_img_data[i];
            nw->target = nw->MNIST_target[i];

            forward_prop_simd_para(nw);
        }
        end = omp_get_wtime();
        fprintf(fp, " %.4f", ((end - start)/ epoch_size));
    }

    fprintf(fp, "\n");
    fclose(fp);
}