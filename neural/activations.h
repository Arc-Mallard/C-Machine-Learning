#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "../matrix/matrix.h"

float sigmoid(float x);

float sigmoidDx(float x);

float swiglu(float f, float g);

float swigluDx(float f, float g);

float swigluDg(float f);

float max(Matrix m);

Matrix softmax(Matrix m);

#endif
