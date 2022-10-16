#include "layer.h"
#include <stdlib.h>
#include <stdio.h>


struct Layer* make_layer(int size) {

    struct Layer* layer = (struct Layer*)malloc(sizeof(struct Layer));
    if(layer == NULL) {
        printf("Error: failed to allocate memory for Layer");
        exit(-1);
    }

    layer->a = (float*)calloc(size, sizeof(float));
    if(layer->a == NULL) {
        printf("Error: failed to allocate memmory for array in Layer");
        exit(-1);
    }

    layer->size = size;

    return layer;
}

struct Layer* make_layer_zero(int size) {
    
    struct Layer* layer = (struct Layer*)malloc(sizeof(struct Layer));
    if(layer == NULL) {
        printf("Error: failed to allocate memory for Layer");
        exit(-1);
    }

    layer->a = (float*)calloc(size, sizeof(float));
    if(layer->a == NULL) {
        printf("Error: failed to allocate memmory for array in Layer");
        exit(-1);
    }

    layer->size = size;

    return layer;
}

void delete_layer(struct Layer* l) {
    free(l->a);
    free(l);
}

void display_layer(struct Layer* layer) {
    for(int i = 0; i < layer->size; i++) {
        printf("%f ", layer->a[i]);
    }
    printf("\n");
}

void display_layer_for_testing(struct Layer* layer) {
    float best = 0.0;
    int number = 0;

    for(int i = 0; i < layer->size; i++) {
        if(layer->a[i] > best) {
            best = layer->a[i];
            number = i;
        }
    }
    printf("prediction: %d,  %.2f%% ", number, (best * 100));
    printf("\n");
}


int get_prediction(struct Layer* layer) {

    float best = 0.0;
    int number = 0;

    for(int i = 0; i < layer->size; i++) {
        if(layer->a[i] > best) {
            best = layer->a[i];
            number = i;
        }
    }

    return number;
}



int get_target(struct Layer* layer) {

    for(int i = 0; i < layer->size; i++) {
        if(layer->a[i] == 1.0) {
            return i;
        }
    }

    return 0;
}