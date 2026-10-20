#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H
#include "../matrix/matrix.h"

typedef struct {
	int layers; int* size; Matrix* weights;
	// These matrices below are actually all vectors
	Matrix* biases; Matrix* activations; Matrix* preactivations;
} Network;

// ALL matrices below should be treated as vectors
Network init(int layers, int* size, float deviation);

Network initEmptyNet(int layers, int* size);

Matrix forward(Network net, Matrix input);

Network backward(Network net, Matrix output, Matrix expected);

Network train(Network net, Matrix inputs[], Matrix expected[], int length, int epochs);

Matrix predict(Network net, Matrix input);

void freeNet(Network* net);

void printNet(Network net);

unsigned char* serializeNetwork(Network n);

Network deserializeNetwork(unsigned char* m);

#endif
