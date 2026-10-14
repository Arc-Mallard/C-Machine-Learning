#include "../matrix/matrix.h"

typedef struct {
	int* size; Matrix* weights;
	// These matrices are actually all vectors
	Matrix* biases; Matrix* activations; Matrix* preactivations;
} Network;

// ALL matrices below should be treated as vectors
Network init(Network net);

Matrix forward(Matrix input);

Network backward(Network net, Matrix output, Matrix expected);

Network train(Network net, Matrix inputs[], Matrix expected[], int epochs);

Matrix predict(Network net, Matrix input);

unsigned char* serializeNetwork(Network n);

Network deserializeNetwork(unsigned char* m);
