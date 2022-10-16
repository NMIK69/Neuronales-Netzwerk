#include <stdio.h>
#include "network.h"
#include "layer.h"
#include "matrix.h"
#include "util.h"
#include "network_learn_seq.h"

void update_weights_seq(struct Network* nw, float* learning_rate) {
    for(int i = 0; i < (nw->layer_num - 1); i++) {
        matrix_update(nw->weights[i], nw->delta_weights[i], (*learning_rate));
    }
}

void fp_matrix_mult_seq(struct Network* nw, struct Matrix* m, struct Layer* l_prev, struct Layer* l) {

    float temp = 0.0;
    for(int i = 0; i < m->H; i++) {
        temp = 0.0;
        for(int j = 0; j < m->W; j++) {
            temp += l_prev->a[j] * m->matrix[i][j];
        }
        l->a[i] = sigmoid_function(&temp);
    }
}


void forward_prop_seq(struct Network* nw) {
    
    for(int i = 0; i < nw->layer_num - 1; i++) {
        fp_matrix_mult_seq(nw, nw->weights[i], nw->layers[i], nw->layers[i + 1]);
    }

}


void backwards_prop_output_layer_seq(struct Network* nw) {

    int out_layer_index = nw->layer_num - 1;
    int hidden_layer_index = nw->layer_num - 2;

    int out_layer_size = nw->layer_sizes[out_layer_index];
    int hidden_layer_size = nw->layer_sizes[hidden_layer_index];

    struct Layer* out_layer = nw->layers[out_layer_index];
    struct Layer* hidden_layer = nw->layers[hidden_layer_index];

    struct Matrix* d_weight = nw->delta_weights[out_layer_index - 1];

    struct Layer* current_comp = nw->computations[nw->computation_index];
    nw->computation_index += 1;

    

    for(int k = 0; k < out_layer_size; k++) {
    
        current_comp->a[k] = (out_layer->a[k] - nw->target->a[k]) * (out_layer->a[k] * (1 - out_layer->a[k]));
        

        for(int j = 0; j < hidden_layer_size; j++) {
            d_weight->matrix[k][j] = hidden_layer->a[j] * current_comp->a[k];
            
        }
    }
}



void backwards_prop_hidden_layer_seq(struct Network* nw, int current_layer_index) {

    // prev layer is the layer to the left, next layer the one to the right
    // prev layer <--- current layer ----> next layer
    // so in a 3 layer network prev would be input, current hidden and next output layer

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


    float temp = 0.0;

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


void backwards_prop_seq(struct Network* nw) {

    backwards_prop_output_layer_seq(nw);

    int hidden_layer_index = nw->layer_num - 2;

    for(int i = hidden_layer_index; i > 0; i--) {
        backwards_prop_hidden_layer_seq(nw, i);
    }


    nw->computation_index = 0;
}