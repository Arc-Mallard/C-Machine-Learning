// https://chsafouane.github.io/posts/why-swiglu-works/index.html
#include "../matrix/matrix.h"
#include <math.h>

float sigmoid(float x) {
	return 1 / (1.0 + expf(-x));
}

float swiglu(float f, float g) {
	return sigmoid(f) * f * g;
}

float swigluDx(float f, float g) {
	float s = sigmoid(f);
	return g * (s + f * s * (1.0 - s));
}

float swigluDg(float f) {
	return f * sigmoid(f);
}
