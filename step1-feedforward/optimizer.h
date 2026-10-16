#include "../matrix/matrix.h"
#include <math.h>

Matrix updateWeights(Matrix weights, Matrix gradient);

// Treat the matrices below as vectors
Matrix updateBiases(Matrix biases, Matrix delta);
