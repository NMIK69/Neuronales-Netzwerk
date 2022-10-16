#pragma once


struct Matrix {

    int W;
    int H;
    float** matrix;

};


struct Matrix* make_matrix(int Height, int Width);
void delete_matrix(struct Matrix* matrix);
void fill_matrix(struct Matrix* matrix, float value);
float uniform_distribution(float low, float high);
void randomize_matrix(struct Matrix* matrix, int n);
void display_matrix(struct Matrix* matrix);
void matrix_update(struct Matrix* m1, struct Matrix* m2, float alpha);
void matrix_tanspose(struct Matrix* src, struct Matrix* dest);