#ifndef NORMAL_H
#define NORMAL_H

#include "../matrix/matrix.h"

float gaussianNoise(float mean, float deviation);

Matrix initNormal(int w, int h, float mean, float deviation);

#endif
