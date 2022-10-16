#include "network.c"
#include "layer.c"
#include "matrix.c"
#include "network_learn_seq.c"
#include "network_learn_para.c"
#include "network_learn_omp_for.c"
#include "network_learn_simd_para.c"
#include "network_learn_simd_seq.c"
#include "util.c"
#include <assert.h>


static void test_seq() {
    int layer_sizes[3] = {3,2,1};
    struct Network* nw = make_network(3, layer_sizes, 1, 1);

    struct Matrix* w1 = nw->weights[0];
    struct Matrix* w2 = nw->weights[1];
    
    struct Layer* l1 = nw->layers[0];
    l1->a[0] = 0.1;
    l1->a[1] = 0.2;
    l1->a[2] = 0.3;

    struct Layer* t1 = nw->target;
    t1->a[0] = 0;

    w1->matrix[0][0] = 0.1;
    w1->matrix[0][1] = 0.2;
    w1->matrix[0][2] = 0.3;

    w1->matrix[1][0] = 0.4;
    w1->matrix[1][1] = 0.5;
    w1->matrix[1][2] = 0.6;

    w2->matrix[0][0] = 0.7;
    w2->matrix[0][1] = 0.8;


    float lr = 1.0;
    forward_prop_seq(nw);   
    backwards_prop_seq(nw);
    update_weights_seq(nw, &lr);

    int temp[2][3];
    for(int i = 0; i < w1->H; i++) {
        for(int j = 0; j < w1->W; j++) {
            temp[i][j] = w1->matrix[i][j] * 100000;
        }
    } 

    assert(temp[0][0] == 9743);
    assert(temp[0][1] == 19487);
    assert(temp[0][2] == 29231);

    assert(temp[1][0] == 39713);
    assert(temp[1][1] == 49426);
    assert(temp[1][2] == 59139);

    delete_network(nw);
}

static void test_para_omp_for() {
    int layer_sizes[3] = {3,2,1};
    struct Network* nw = make_network(3, layer_sizes, 1, 1);

    struct Matrix* w1 = nw->weights[0];
    struct Matrix* w2 = nw->weights[1];
    
    struct Layer* l1 = nw->layers[0];
    l1->a[0] = 0.1;
    l1->a[1] = 0.2;
    l1->a[2] = 0.3;

    struct Layer* t1 = nw->target;
    t1->a[0] = 0;

    w1->matrix[0][0] = 0.1;
    w1->matrix[0][1] = 0.2;
    w1->matrix[0][2] = 0.3;

    w1->matrix[1][0] = 0.4;
    w1->matrix[1][1] = 0.5;
    w1->matrix[1][2] = 0.6;

    w2->matrix[0][0] = 0.7;
    w2->matrix[0][1] = 0.8;


    float lr = 1.0;
    forward_prop_omp_for(nw);   
    backwards_prop_omp_for(nw);
    update_weights_seq(nw, &lr);

    int temp[2][3];
    for(int i = 0; i < w1->H; i++) {
        for(int j = 0; j < w1->W; j++) {
            temp[i][j] = w1->matrix[i][j] * 100000;
        }
    } 

    assert(temp[0][0] == 9743);
    assert(temp[0][1] == 19487);
    assert(temp[0][2] == 29231);

    assert(temp[1][0] == 39713);
    assert(temp[1][1] == 49426);
    assert(temp[1][2] == 59139);

    delete_network(nw);
}

static void test_para_simd() {
    int layer_sizes[3] = {3,2,1};
    struct Network* nw = make_network(3, layer_sizes, 1, 1);

    struct Matrix* w1 = nw->weights[0];
    struct Matrix* w2 = nw->weights[1];
    
    struct Layer* l1 = nw->layers[0];
    l1->a[0] = 0.1;
    l1->a[1] = 0.2;
    l1->a[2] = 0.3;

    struct Layer* t1 = nw->target;
    t1->a[0] = 0;

    w1->matrix[0][0] = 0.1;
    w1->matrix[0][1] = 0.2;
    w1->matrix[0][2] = 0.3;

    w1->matrix[1][0] = 0.4;
    w1->matrix[1][1] = 0.5;
    w1->matrix[1][2] = 0.6;

    w2->matrix[0][0] = 0.7;
    w2->matrix[0][1] = 0.8;


    float lr = 1.0;
    forward_prop_simd_para(nw);   
    backwards_prop_simd_para(nw);
    update_weights_seq(nw, &lr);

    int temp[2][3];
    for(int i = 0; i < w1->H; i++) {
        for(int j = 0; j < w1->W; j++) {
            temp[i][j] = w1->matrix[i][j] * 100000;
        }
    } 

    assert(temp[0][0] == 9743);
    assert(temp[0][1] == 19487);
    assert(temp[0][2] == 29231);

    assert(temp[1][0] == 39713);
    assert(temp[1][1] == 49426);
    assert(temp[1][2] == 59139);

    delete_network(nw);
}

static void test_seq_simd() {
    int layer_sizes[3] = {3,2,1};
    struct Network* nw = make_network(3, layer_sizes, 1, 1);

    struct Matrix* w1 = nw->weights[0];
    struct Matrix* w2 = nw->weights[1];
    
    struct Layer* l1 = nw->layers[0];
    l1->a[0] = 0.1;
    l1->a[1] = 0.2;
    l1->a[2] = 0.3;

    struct Layer* t1 = nw->target;
    t1->a[0] = 0;

    w1->matrix[0][0] = 0.1;
    w1->matrix[0][1] = 0.2;
    w1->matrix[0][2] = 0.3;

    w1->matrix[1][0] = 0.4;
    w1->matrix[1][1] = 0.5;
    w1->matrix[1][2] = 0.6;

    w2->matrix[0][0] = 0.7;
    w2->matrix[0][1] = 0.8;


    float lr = 1.0;
    forward_prop_simd_seq(nw);   
    backwards_prop_simd_seq(nw);
    update_weights_seq(nw, &lr);

    int temp[2][3];
    for(int i = 0; i < w1->H; i++) {
        for(int j = 0; j < w1->W; j++) {
            temp[i][j] = w1->matrix[i][j] * 100000;
        }
    } 

    assert(temp[0][0] == 9743);
    assert(temp[0][1] == 19487);
    assert(temp[0][2] == 29231);

    assert(temp[1][0] == 39713);
    assert(temp[1][1] == 49426);
    assert(temp[1][2] == 59139);

    delete_network(nw);
}

int main() {

    test_seq();
    test_para_omp_for();
    test_para_simd();
    test_seq_simd();

    return 0;
}