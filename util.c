#include "util.h"
#include <math.h>

float sigmoid_function(float* net) {
    float temp = 1 + exp( (-1 * (*net)) );
    return (1.0/temp);
}