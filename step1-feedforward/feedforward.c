// ALL matrices below should be treated as vectors
#include "feedforward.h"

Matrix forward(Matrix input) {
	return (Matrix) {0};
}

Network backward(Network net, Matrix output, Matrix expected) {
	return (Network) {0};
}

Network train(Network net, Matrix inputs[], Matrix expected[], int epochs) {
	return (Network) {0};
}

Matrix predict(Network net, Matrix input) {
	return (Matrix) {0};
}

unsigned char* serializeNetwork(Network n) {
	unsigned char weights[0];
	return weights;
}

Network deserializeNetwork(unsigned char* m) {
	return (Network) {0};
}
