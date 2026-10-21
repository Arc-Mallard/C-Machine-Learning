// ALL matrices below should be treated as vectors
#include "feedforward.h"
#include "adamw.h"
#include "normal.h"
#include "sparce.h"
#include "activations.h"

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
	s[0] = size[0];
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

Network backward(AdamwState* adamw, Network net, Matrix output, Matrix expected) {
    Matrix delta = sccDerivitive(output, expected);
    //AdamwState adamw = adamwInit(net, 0.01, 0.1);

    for (int i = 0; i < delta.n; i++ ) {
        delta.d[i] *= sigmoidDx(net.preactivations[net.layers - 2].d[i]);
    }

    Matrix weightGrad[net.layers - 1], biasGrad[net.layers - 1];

    for (int i = net.layers - 2; i >= 0; i--) {
        weightGrad[i] = multiply(delta, transpose(net.activations[i]));
        biasGrad[i] = delta;
        if (i != 0) {
            delta = multiply(transpose(net.weights[i]), delta);
            for (int j = 0; j < delta.n; j++) {
                delta.d[j] *= sigmoidDx(net.preactivations[i - 1].d[j]);
            }
        }
    }

    Network optimized = adamwOptimize(adamw, net, weightGrad, biasGrad);

	return optimized;
}

Network train(Network net, Matrix inputs[], Matrix expected[], int length, int epochs) {
    AdamwState adamw = adamwInit(net, 0.01, 0.1);
	Network trained, current; Matrix prediction;
	for (int i = 0; i < epochs; i++) {
		for (int j = 0; j < length; j++) {
			prediction = forward(net, inputs[j]);
			current = backward(&adamw, net, prediction, expected[j]);
			trained = current; free(prediction.d);
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

void freeNet(Network* net) {
	freeMatrixArr(net -> weights, net -> layers - 1);
	freeMatrixArr(net -> biases, net -> layers - 1);
	freeMatrixArr(net -> activations, net -> layers);
	freeMatrixArr(net -> preactivations, net -> layers - 1);
}

void printNet(Network net) {
    printf("Network:\n");
    printf("Weights:\n");
    for (int i = 0; i < net.layers - 1; i++) {
        printf("%d:\n", i);
        print(net.weights[i]);
    }
    printf("\n");
}

unsigned char* serializeNetwork(Network n) {
	unsigned char* temp;
	return temp;
}

Network deserializeNetwork(unsigned char* m) {
	return (Network) {0};
}
