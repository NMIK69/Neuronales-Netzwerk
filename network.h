#pragma once

struct Network{

    int layer_num;
    int* layer_sizes;
    float error;

    int training_data_size;
    int testing_data_size;

    int training_index;
    int testing_index;
    int computation_index;
    
    long long number_of_nodes;
    long long number_of_weights;
    long long fp_itterations;
    long long bp_itterations;

    struct Matrix** weights;
    struct Matrix** delta_weights;

    struct Layer** layers;
    struct Layer* l1;

    struct Layer* target;
    struct Layer** computations;


    struct Layer** MNIST_target;
    struct Layer** MNIST_img_data;

    struct Layer** MNIST_testing_target;
    struct Layer** Mnist_testing_img_date;
};


struct Network* make_network(int layer_num, int* layer_sizes, int training_data_size, int testing_data_size);
void delete_network(struct Network* nw);

void make_network_information(struct Network* nw);
void display_network_information(struct Network* nw);


int read_in_MNIST_img_data(struct Network* nw);
int read_in_MNIST_target(struct Network* nw);

float get_network_performance(struct Network* nw);



void update_weights(struct Network* nw, float* learning_rate);

void train_network_para(struct Network* nw, float learning_rate);
void train_network_seq(struct Network* nw, float learning_rate);
void train_network_simd_seq(struct Network* nw, float learning_rate);
void train_network_smid_para(struct Network* nw, float learning_rate);
void train_network_omp_for(struct Network* nw, float learning_rate);


void compare_and_save(struct Network* nw, float learning_rate, char* filename, int imp, char* file_mode);
void compare_training(struct Network* nw, float learning_rate, int imp);
void compare_fp_only(struct Network* nw, int imp);
void compare_fp_only_and_save(struct Network* nw, char* filename, int imp, char* file_mode);