// ALL matrices below should be treated as vectors
#include "feedforward.h"

Network train(Matrix inputs[], Matrix expected[], int epochs) {
	return (Network) {0};
}

Matrix predict(Matrix input) {
	return (Matrix) {0};
}

unsigned char* serializeNetwork(Network n) {
	unsigned char weights[0];
	return weights;
}

Network deserializeNetwork(unsigned char* m) {
	return (Network) {0};
}
