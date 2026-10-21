#ifndef SPARCE_H
#define SPARCE_H

#include "../matrix/matrix.h"

// ALL matricies in this file should be treated as vectors

float sccFunction(Matrix output, Matrix target);

Matrix sccDerivitive(Matrix output, Matrix target);

float mseFunction(Matrix output, Matrix target);
Matrix mseDerivative(Matrix output, Matrix target);

#endif
