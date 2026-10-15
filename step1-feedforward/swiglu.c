// https://chsafouane.github.io/posts/why-swiglu-works/index.html
#include "../matrix/matrix.h"
#include <math.h>

float swish(float x) {
	return x / (1.0+expf(-x));
}

float swiglu(float f, float g) {
	return swish(f) * f * g;
}

float swigluDx(float f, float g) {
	float s = swish(f);
	return g * (s + f * s * (1.0 - s));
}

float swigluDg(float f) {
	return f * swish(f);
}
