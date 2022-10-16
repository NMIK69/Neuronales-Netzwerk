#include "network.h"
#include "layer.h"
#include "matrix.h"
#include "util.h"
#include "network_learn_simd_para.h"
#include <omp.h>

void update_weights_simd_para(struct Network* nw, float* learning_rate) {
    #pragma omp parallel for
    for(int i = 0; i < (nw->layer_num - 1); i++) {
        matrix_update(nw->weights[i], nw->delta_weights[i], (*learning_rate));
    }
}

float dot_product_simd_para(float* l_prev, float* weight, int size) {
    float sum = 0.0;

    #pragma omp simd reduction(+:sum)
    for(int i = 0; i < size; i++) {
        sum += *l_prev++ * *weight++;
    }

    return sum;
}

void fp_matrix_mult_simd_para(struct Network* nw, struct Matrix* m, struct Layer* l_prev, struct Layer* l) {

    #pragma omp parallel 
    {   
        float temp;
        #pragma omp for
        for(int i = 0; i < m->H; i++) {
            temp = dot_product_simd_para(l_prev->a, m->matrix[i], m->W);
            l->a[i] = sigmoid_function(&temp);
        }
    }
}


void forward_prop_simd_para(struct Network* nw) {
    
    for(int i = 0; i < nw->layer_num - 1; i++) {
        fp_matrix_mult_simd_para(nw, nw->weights[i], nw->layers[i], nw->layers[i + 1]);
    }

}


void backwards_prop_output_layer_simd_para(struct Network* nw) {

    int out_layer_index = nw->layer_num - 1;
    int hidden_layer_index = nw->layer_num - 2;

    int out_layer_size = nw->layer_sizes[out_layer_index];
    int hidden_layer_size = nw->layer_sizes[hidden_layer_index];

    struct Layer* out_layer = nw->layers[out_layer_index];
    struct Layer* hidden_layer = nw->layers[hidden_layer_index];

    struct Matrix* d_weight = nw->delta_weights[out_layer_index - 1];

    struct Layer* current_comp = nw->computations[nw->computation_index];
    nw->computation_index += 1;

    
    
    #pragma omp for simd
    for(int k = 0; k < out_layer_size; k++) {
    
        current_comp->a[k] = (out_layer->a[k] - nw->target->a[k]) * (out_layer->a[k] * (1 - out_layer->a[k]));
        

        for(int j = 0; j < hidden_layer_size; j++) {
            d_weight->matrix[k][j] = hidden_layer->a[j] * current_comp->a[k];
            
        }
    }
}



void backwards_prop_hidden_layer_simd_para(struct Network* nw, int current_layer_index) {

    // prev layer is the layer to the left, next layer the one to the right
    // prev layer <--- current layer ----> next layer
    // in a 3 layer network prev would be input, current hidden and next output layer

    int current_layer_size = nw->layer_sizes[current_layer_index];
    int prev_layer_size = nw->layer_sizes[current_layer_index - 1];
    int next_layer_size = nw->layer_sizes[current_layer_index + 1];

    struct Layer* current_layer = nw->layers[current_layer_index];
    struct Layer* prev_layer = nw->layers[current_layer_index - 1];

    struct Matrix* weight = nw->weights[current_layer_index];
    struct Matrix* d_weight = nw->delta_weights[current_layer_index - 1];

    struct Layer* prev_comp = nw->computations[nw->computation_index - 1];
    struct Layer* current_comp = nw->computations[nw->computation_index];

    nw->computation_index += 1;


    #pragma omp parallel 
    {
        float temp = 0.0;
        
        #pragma omp for simd
        for(int j = 0; j < current_layer_size; j++) {


            for(int k = 0; k < next_layer_size; k++) {
                temp += (prev_comp->a[k] * weight->matrix[k][j]);
            }

            current_comp->a[j] = (temp * (current_layer->a[j]) * (1 - current_layer->a[j]));


            for(int i = 0; i < prev_layer_size; i++) {
                d_weight->matrix[j][i] = (current_comp->a[j] * prev_layer->a[i]);

            }
            temp = 0.0;

        }
    }
}




void backwards_prop_simd_para(struct Network* nw) {

    backwards_prop_output_layer_simd_para(nw);

    int hidden_layer_index = nw->layer_num - 2;

    for(int i = hidden_layer_index; i > 0; i--) {
        backwards_prop_hidden_layer_simd_para(nw, i);
    }

    nw->computation_index = 0;
}