#pragma once

void fp_matrix_mult_simd_seq(struct Network* nw, struct Matrix* m, struct Layer* l_prev, struct Layer* l);
void forward_prop_simd_seq(struct Network* nw);


void backwards_prop_simd_seq(struct Network* nw);
void backwards_prop_output_layer_simd_seq(struct Network* nw);
void backwards_prop_hidden_layer_simd_seq(struct Network* nw, int current_layer_index);
void update_weights_simd_seq(struct Network* nw, float* learning_rate);
float dot_product(float* l_prev, float* weight, int size);