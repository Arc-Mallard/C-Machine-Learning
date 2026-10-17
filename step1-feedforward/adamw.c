// https://adelbennaceur.github.io/posts/adam_vs_adamw/

#include "optimizer.h"

#define EPSILON 10e-6f

typedef struct {
	float learningRate; float wghtDecay;
	float frstDecay; float scndDecay; int step;
} AdamParams;

typedef struct {
	AdamParams* params;
	Matrix frstMoment;
	Matrix scndMoment;
} AdamState;

AdamState adamInit(Network net, float learningRate, float weightDecay) {
	AdamState s = {
		malloc(sizeof(AdamParams) * net.layers),
		(Matrix) {0}, (Matrix) {0}
	}; AdamParams p;
	for (int i = 0; i < net.weights->n; i++) {
		p = (AdamParams){learningRate, weightDecay, 0.9, 0.999, 0};
	}

	return s;
}

Network adamwOptimize(AdamState* state, Network net) {
	return net;
}





