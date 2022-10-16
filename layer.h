#pragma once

struct Layer{

    int size;
    float* a;

};

struct Layer* make_layer(int size);
void delete_layer(struct Layer* l);
struct Layer* make_layer_zero(int size);
void display_layer(struct Layer* layer);
void display_layer_for_testing(struct Layer* layer);
int get_prediction(struct Layer* layer);
int get_target(struct Layer* layer);