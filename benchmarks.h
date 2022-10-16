#pragma once

void benchmark_save(int imp, int layer_num, int* layer_sizes, int epoch_size, char* filename, char* filemode);
void benchmark_display(int imp, int layer_num, int* layer_sizes, int epoch_size);

void benchmark_fp_only_display(int imp, int layer_num, int* layer_sizes, int epoch_size);
void benchmark_fp_only_save(int imp, int layer_num, int* layer_sizes, int epoch_size, char* filename, char* filemode);

void benchmark_autogenerate(int layer_num, int ls, int step, int N, char* filename);