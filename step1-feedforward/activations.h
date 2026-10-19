#include "../matrix/matrix.h"
#include <math.h>

float sigmoid(float x);

float sigmoidDx(float x);

float swiglu(float f, float g);

float swigluDx(float f, float g);

float swigluDg(float f);

float max(Matrix m);

Matrix softmax(Matrix m);
