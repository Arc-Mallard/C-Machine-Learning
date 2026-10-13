#include "../matrix/matrix.h"

typedef struct {
	int* size; Matrix* weights;
	// These matrices are actually all vectors
	Matrix* biases; Matrix* activations; Matrix* preactivations;
} Network;

// ALL matrices below should be treated as vectors

Network train(Matrix inputs[], Matrix expected[], int epochs);

Matrix predict(Matrix input);

unsigned char* serializeNetwork(Network n);

Network deserializeNetwork(unsigned char* m);
