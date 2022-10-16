#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"

struct Matrix* make_matrix(int Height, int Width) {

    struct Matrix* matrix = (struct Matrix*)malloc(sizeof(struct Matrix));
    if(matrix == NULL) {
        printf("Error: failed to allocate memory for Matrix\n");
        exit(-2);
    }

    matrix->H = Height;
    matrix->W = Width;

    matrix->matrix = (float**)malloc(Height * sizeof(float*));
    if(matrix->matrix == NULL) {
        printf("Error: failed to allocate memory for arr** in Matrix\n");
        exit(-2);
    }

    for(int i = 0; i < Height; i++) {
        matrix->matrix[i] = (float*)malloc(Width * sizeof(float));
        if(matrix->matrix[i] == NULL) {
            printf("Error: failed to allocate memory for arr* in Matrix->matrix\n");
            exit(-2);
        }
    }

    return matrix;
}

void delete_matrix(struct Matrix* matrix) {

    for(int i = 0; i < matrix->H; i++) {
        free(matrix->matrix[i]);
    }

    free(matrix->matrix);
    free(matrix);
}

void fill_matrix(struct Matrix* matrix, float value) {

    for(int i = 0; i < matrix->H; i++) {
        for(int j = 0; j < matrix->W; j++) {
            matrix->matrix[i][j] = value;
        }
    }

}

float uniform_distribution(float low, float high) {
	float difference = high - low; // The difference between the two
	int scale = 10000;
	int scaled_difference = (int)(difference * scale);
	return low + (1.0 * (rand() % scaled_difference) / scale);
}


void randomize_matrix(struct Matrix* matrix, int n) {
    float low = -1.0;
	float high = 1.0;
    for(int i = 0; i < matrix->H; i++) {
        for(int j = 0; j < matrix->W; j++) {
            matrix->matrix[i][j] = uniform_distribution(low, high);
        }
    }
}

void display_matrix(struct Matrix* matrix) {

    for(int i = 0; i < matrix->H; i++) {
        for(int j = 0; j < matrix->W; j++) {
            printf("%f ", matrix->matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


void matrix_update(struct Matrix* m1, struct Matrix* m2, float alpha) {

    if(m1->H != m2->H || m1->W != m2->W) {
        printf("Error in matrix_update: matricies dont match\n");
        return;
    }

    for(int i = 0; i < m1->H; i++) {
        for(int j = 0; j < m1->W; j++) {
            m1->matrix[i][j] = m1->matrix[i][j] - (alpha * m2->matrix[i][j]);
        }
    }
}


void matrix_tanspose(struct Matrix* src, struct Matrix* dest) {
    for(int i = 0; i < src->H; i++) {
        for(int j = 0; j < src->W; j++) {
            dest->matrix[j][i] = src->matrix[i][j];
        }
    }
}