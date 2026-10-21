// https://www.geeksforgeeks.org/machine-learning/what-is-sparse-categorical-crossentropy/

#include "costs.h"
#include <math.h>

// ALL matrices in this file should be treated as vectors

float sccFunction(Matrix output, Matrix target) {
    int samples = target.n;
    int classes = output.n / target.n;
    double total = 0.0;

    for (int i = 0; i < samples; i++) {
        int label = (int)target.d[i];

        if (label < 0 || label >= classes)
            continue;

        double p = output.d[i * classes + label];

        if (p < EPSILON)
            p = EPSILON;
        if (p > 1.0 - EPSILON)
            p = 1.0 - EPSILON;

        total += -log(p);
    }

    return (float)(total / samples);
}

Matrix sccDerivitive(Matrix output, Matrix target) {
	int k, label, samples = target.n, classes = output.n / target.n;
	Matrix m = initEmptyMatrix(1, output.n);
	for (int i = 0; i < samples; i++) {
		label = (int) target.d[i];
		for (int j = 0; j < classes; j++) {
			k = i * classes + j;
			m.d[k] = (j == label? output.d[k] - 1.0f : output.d[k]) / samples;
		}
	} return m;
}

float mseFunction(Matrix output, Matrix target) {
    float diff, cost = 0.0;
    for (int i = 0; i < output.n; i++) {
        diff = output.d[i] - target.d[i];
        cost += diff * diff;
    }
    return cost / output.n;
}

Matrix mseDerivative(Matrix output, Matrix target) {
    Matrix m = initEmptyMatrix(1, output.n);
    for (int i = 0; i < m.n; i++) {
        m.d[i] = 2.0 * (output.d[i] - target.d[i]) / output.n;
    }

    return m;
}
