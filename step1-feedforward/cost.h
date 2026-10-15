#include "../matrix/matrix.h"
#include <math.h>

#define EPSILON 1e-7

// ALL matricies in this file should be treated as vectors

float cstFunction(Matrix output, Matrix target);

Matrix cstDerivitive(Matrix output, Matrix target);
