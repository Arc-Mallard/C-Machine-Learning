#include "normal.h"

float gaussianNoise(float mean, float deviation) {
    float u1 = ((float)rand() + 1.0) / ((float)RAND_MAX + 2.0);
    float u2 = ((float)rand() + 1.0) / ((float)RAND_MAX + 2.0);

    float z = deviation * sqrt(-2 * log(u1)) * cos(2 * M_PI * u2) + mean;

    return z;
}

Matrix initNormal(int w, int h, int n, float mean, float deviation) {
    Matrix res = {0};

    res.w = w;
    res.h = h;
    res.n = n;
    res.d = malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        res.d[i] = gaussianNoise(mean, deviation);
    }
	return res;
}
