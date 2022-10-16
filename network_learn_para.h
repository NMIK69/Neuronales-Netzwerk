#pragma once

void fp_matrix_mult_para(struct Network* nw, struct Matrix* m, struct Layer* l_prev, struct Layer* l);
void forward_prop_para(struct Network* nw);


void backwards_prop_para(struct Network* nw);
void backwards_prop_output_layer_para(struct Network* nw);
void backwards_prop_hidden_layer_para(struct Network* nw, int current_layer_index);
void update_weights_para(struct Network* nw, float* learning_rate);