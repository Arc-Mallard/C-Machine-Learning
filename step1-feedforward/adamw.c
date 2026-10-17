// heavily "inspired" by https://github.com/loretoparisi/adamw.c/blob/main/adamw.c

#include "optimizer.h"

#define EPSILON 10e-6f

typedef struct {
	float learningRate; float wghtDecay;
	float frstDecay; float scndDecay; int step;
} AdamParams;

typedef struct {
	AdamParams params;
	Matrix* frstMomentWght; Matrix* scndMomentWght;
	Matrix* frstMomentBias; Matrix* scndMomentBias;
} AdamState;

AdamState adamInit(Network net, float learningRate, float weightDecay) {
	AdamState s = {
		(AdamParams) {learningRate, weightDecay, 0.9, 0.999, 0},
		malloc(sizeof(Matrix) * net.layers), malloc(sizeof(Matrix) * net.layers),
		malloc(sizeof(Matrix) * net.layers), malloc(sizeof(Matrix) * net.layers)
	};  // Who needs error handling anyways?
	for (int i = 0; i < net.layers; i++) {
		// Initialize the moment for the network weights
		s.frstMomentWght[i].w = net.weights[i].w; s.scndMomentWght[i].w = net.weights[i].w;
		s.frstMomentWght[i].h = net.weights[i].h; s.scndMomentWght[i].h = net.weights[i].h;
		s.frstMomentWght[i].n = net.weights[i].n; s.scndMomentWght[i].n = net.weights[i].n;
		// Initialize the moment for the network biases
		s.frstMomentBias[i].w = net.biases[i].w; s.scndMomentBias[i].w = net.biases[i].w;
		s.frstMomentBias[i].h = net.biases[i].h; s.scndMomentBias[i].h = net.biases[i].h;
		s.frstMomentBias[i].n = net.biases[i].n; s.scndMomentBias[i].n = net.biases[i].n;
		// Allocate memory for the moment of the network weights
		s.frstMomentWght[i].d = malloc(sizeof(float)*net.weights[i].n);
		s.scndMomentWght[i].d = malloc(sizeof(float)*net.weights[i].n);
		// Allocate memory for the moment of the network biases
		s.frstMomentBias[i].d = malloc(sizeof(float)*net.biases[i].n);
		s.scndMomentBias[i].d = malloc(sizeof(float)*net.biases[i].n);
		// Set garbage memory to 0 for weights and biases
		for (int j = 0; j < net.weights[i].n; j++) {s.frstMomentWght[i].d[j] = 0.0; s.scndMomentWght[i].d[j] = 0.0;}
		for (int j = 0; j < net.biases[i].n; j++) {s.frstMomentBias[i].d[j] = 0.0; s.scndMomentBias[i].d[j] = 0.0;}
	} return s;
}

void adamFree(AdamState* state, Network net) {
	for (int i = 0; i < net.layers; i++) {
		free(state -> frstMomentWght); free(state -> scndMomentWght);
		free(state -> frstMomentBias); free(state -> scndMomentBias);
	}  // Free the matrices after freeing their data array
	free(state -> frstMomentWght); free(state -> scndMomentWght);
	free(state -> frstMomentBias); free(state -> scndMomentBias);
}

Network adamwOptimize(AdamState* state, Network net, Matrix gradient) {
	state -> params.step++;



	return net;
}





