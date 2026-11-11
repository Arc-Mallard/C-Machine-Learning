#ifndef TYPES_H
#define TYPES_H

typedef struct {
	int w, h, n;
	float* d;
} Matrix;

typedef struct {
	float learningRate; float wghtDecay;
	float frstDecay; float scndDecay; int step;
} AdamwParams;

typedef struct {
	AdamwParams params;
	Matrix* frstMomentWght; Matrix* scndMomentWght;
	Matrix* frstMomentBias; Matrix* scndMomentBias;
	// Swiglu additions
	Matrix* frstMomentWght2; Matrix* scndMomentWght2;
	Matrix* frstMomentBias2; Matrix* scndMomentBias2;
} AdamwState;

typedef struct {
	int layers; int* size; Matrix* weights1; Matrix* weights2;
	// These matrices below are actually all vectors
	Matrix* biases1; Matrix* biases2;
	Matrix* activations; Matrix* preactivations; Matrix* gpreactivations;

	AdamwState adamw;
} Network;

#endif
