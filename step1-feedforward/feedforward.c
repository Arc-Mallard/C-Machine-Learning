// ALL matrices below should be treated as vectors
#include "feedforward.h"
#include "normal.h"

Network init(int layers, int* size, float deviation) {
    int layersAdjust = layers - 1;
    Matrix* wghts = calloc(layersAdjust, sizeof(Matrix));
    Matrix* biases = calloc(layersAdjust, sizeof(Matrix));
    Matrix* activations = calloc(layers, sizeof(Matrix));
    Matrix* preactivations = calloc(layersAdjust, sizeof(Matrix));
    activations[0] = initEmptyMatrix(1, size[0]);

    for (int i = 1; i < layers; i++) {
        wghts[i - 1] = initNormal(size[i - 1], size[i], 0.0, deviation);
        biases[i - 1] = initEmptyMatrix(1, size[i]);
        activations[i] = initEmptyMatrix(1, size[i]);
        preactivations[i - 1] = initEmptyMatrix(1, size[i]);
    }

	return (Network) {layers, size, wghts, biases, activations, preactivations};
}

Network initEmptyNet(int layers, int* size) {
	int* s = malloc(sizeof(int) * layers);
	int layersAdjust = layers - 1; Matrix* wghts = malloc(layersAdjust * sizeof(Matrix));
	Matrix* biases = malloc(layersAdjust * sizeof(Matrix)); Matrix* activations = malloc(layers * sizeof(Matrix));
	Matrix* preactivations = malloc(layersAdjust * sizeof(Matrix)); activations[0] = initEmptyMatrix(1, size[0]);
	for (int i = 1; i < layers; i++) {
		wghts[i-1] = initEmptyMatrix(size[i-1], size[i]); biases[i-1] = initEmptyMatrix(1, size[i]);
		activations[i] = initEmptyMatrix(1, size[i]); preactivations[i-1] = initEmptyMatrix(1, size[i]);
		s[i] = size[i];
	} return (Network) {layers, s, wghts, biases, activations, preactivations};
}

Matrix forward(Network net, Matrix input) {
	memcpy(net.activations[0].d, input.d, input.n * sizeof(float));
	Matrix current = copyMatrix(input);
	for (int i = 0; i < net.layers-1; i++) {
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

Network train(Network net, Matrix inputs[], Matrix expected[], int length, int epochs) {
	Network trained, current; Matrix prediction;
	for (int i = 0; i < epochs; i++) {
		for (int j = 0; j < length; j++) {
			prediction = forward(net, inputs[j]);
			current = backward(net, prediction, expected[j]);
			trained = copyNet(current); freeNet(&current);
			free(prediction.d); free(&prediction);
		}
	} return trained;
}

Matrix predict(Network net, Matrix input) {
	Matrix current = copyMatrix(input);
	for (int i = 0; i < net.layers-1; i++) {
		Matrix product = multiply(net.weights[i], current);
		Matrix sum = addMatrix(net.biases[i], product);
		for (int j = 0; j < sum.n; j++) {sum.d[j] = sigmoid(sum.d[j]);}
		free(product.d); free(current.d); current = sum;
	} return current;
}

Network copyNet(Network net) {
	int* s = malloc(sizeof(int) * net.layers);
	int layersAdjust = net.layers - 1; Matrix* wghts = malloc(layersAdjust * sizeof(Matrix));
	Matrix* biases = malloc(layersAdjust * sizeof(Matrix)); Matrix* activations = malloc(net.layers * sizeof(Matrix));
	Matrix* preactivations = malloc(layersAdjust * sizeof(Matrix)); activations[0] = copyMatrix(net.activations[0]);
	for (int i = 1; i < net.layers; i++) {
		s[i] = net.size[i]; wghts[i-1] = copyMatrix(net.weights[i-1]); biases[i-1] = copyMatrix(net.biases[i-1]);
		activations[i] = copyMatrix(net.activations[i-1]); preactivations[i-1] = copyMatrix(net.preactivations[i-1]);
	} return (Network) {net.layers, s, wghts, biases, activations, preactivations};
}

void freeNet(Network* net) {
	freeMatrixArr(net -> weights, net -> layers - 1);
	freeMatrixArr(net -> biases, net -> layers - 1);
	freeMatrixArr(net -> activations, net -> layers);
	freeMatrixArr(net -> preactivations, net -> layers - 1);
	free(net);
}

unsigned char* serializeNetwork(Network n) {
	unsigned char* temp;
	return temp;
}

Network deserializeNetwork(unsigned char* m) {
	return (Network) {0};
}
