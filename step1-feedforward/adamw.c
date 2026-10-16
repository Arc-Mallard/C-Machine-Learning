// https://adelbennaceur.github.io/posts/adam_vs_adamw/

#include "optimizer.h"

#define EPSILON 10e-6f

typedef struct {
	float learningRate; float wghtDecay;
	float frstDecay; float scndDecay; int step;
} AdamState;

AdamState adamInit(float learningRate, float weightDecay) {
	AdamState a = {learningRate, weightDecay, 0.9, 0.999, 0}; return a;
}

Network adamwOptimize(AdamState* state, Matrix* frstMoment, Matrix* scndMoment, Network net) {
	return net;
}





