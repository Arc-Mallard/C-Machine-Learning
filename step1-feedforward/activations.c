#include "activations.h"

float swiglu(float f, float g) {
	return sigmoid(f) * f * g;
}

float swigluDx(float f, float g) {
	float s = sigmoid(f);
	return g * (s + f * s * (1.0f - s));
}

float swigluDg(float f) {
	return f * sigmoid(f);
}

float sigmoid(float x) {
	return 1 / (1.0f + expf(-x));
}

float sigmoidDx(float x) {
	float s = sigmoid(x); return s * (1 - s);
}

Matrix softmax(Matrix m) {
	float ttl = max(m), sum = 0;
	Matrix a = initEmptyMatrix(m.w, m.h);
	for (int i = 0; i < m.n; i++) {
		a.d[i] = expf(m.d[i] - ttl); sum += a.d[i];
	} for (int i = 0; i < m.n; i++) {a.d[i] /= sum;}
	return a;
}

float max(Matrix m) {
	float max = m.d[0];
	for (int i = 1; i < m.n; i++) {
		max = m.d[i] > max? m.d[i] : max;
	} return max;
}
