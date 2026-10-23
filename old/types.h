#ifndef TYPES_H
#define TYPES_H

typedef struct {
	int w, h, n;
	float* d;
} Matrix;

typedef struct {
	int layers; int* size; Matrix* weights;
	// These matrices below are actually all vectors
	Matrix* biases; Matrix* activations; Matrix* preactivations;
} Network;

typedef struct {
	float learningRate; float wghtDecay;
	float frstDecay; float scndDecay; int step;
} AdamwParams;

typedef struct {
	AdamwParams params;
	Matrix* frstMomentWght; Matrix* scndMomentWght;
	Matrix* frstMomentBias; Matrix* scndMomentBias;
} AdamwState;

#endif
