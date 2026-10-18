// ALL matrices below should be treated as vectors
#include "feedforward.h"

Network init(int size[]) {
	return (Network) {0};
}

Matrix forward(Network net, Matrix input) {
	memcpy(net.activations[0].d, input.d, input.n * sizeof(float));
	Matrix current = copyMatrix(input);
	for (int i = 0; i < net.layers; i++) {
		Matrix product = multiply(net.weights[i], current);
		Matrix sum = addMatrix(net.biases[i], product);
		net.preactivations[i] = copyMatrix(sum);
		for (int j = 0; j < sum.n; j++) {
			sum.d[j] = sigmoid(sum.d[j]);
		} net.activations[i+1] = copyMatrix(sum);
		free(product.d); free(current.d); current = sum;
	} return current;
}

Network backward(Network net, Matrix output, Matrix expected) {
	return (Network) {0};
}

Network train(Network net, Matrix inputs[], Matrix expected[], int epochs) {
	return (Network) {0};
}

Matrix predict(Network net, Matrix input) {
	Matrix current = copyMatrix(input);
	for (int i = 0; i < net.layers; i++) {
		Matrix product = multiply(net.weights[i], current);
		Matrix sum = addMatrix(net.biases[i], product);
		for (int j = 0; j < sum.n; j++) {sum.d[j] = sigmoid(sum.d[j]);}
		free(product.d); free(current.d); current = sum;
	} return current;
}

unsigned char* serializeNetwork(Network n) {
	unsigned char* temp;
	return temp;
}

Network deserializeNetwork(unsigned char* m) {
	return (Network) {0};
}
