#ifndef TYPES_OLD_H
#define TYPES_OLD_H

typedef struct {
	int layers; int* size; Matrix* weights;
	// These matrices below are actually all vectors
	Matrix* biases; Matrix* activations; Matrix* preactivations;
} Old_Network;

typedef struct {
	float learningRate; float wghtDecay;
	float frstDecay; float scndDecay; int step;
} Old_AdamwParams;

typedef struct {
	Old_AdamwParams params;
	Matrix* frstMomentWght; Matrix* scndMomentWght;
	Matrix* frstMomentBias; Matrix* scndMomentBias;
} Old_AdamwState;

#endif
