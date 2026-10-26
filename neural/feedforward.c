// ALL matrices below should be treated as vectors
#include "../matrix/matrix.h"
#include "feedforward.h"
#include "adamw.h"
#include "normal.h"
#include "costs.h"
#include "activations.h"

Network init(int layers, int* size, float deviation) {
    int layersAdjust = layers - 1;
    Matrix* wghts1 = calloc(layersAdjust, sizeof(Matrix));
    Matrix* wghts2 = calloc(layersAdjust, sizeof(Matrix));
    Matrix* biases1 = calloc(layersAdjust, sizeof(Matrix));
    Matrix* biases2 = calloc(layersAdjust, sizeof(Matrix));
    Matrix* activations1 = calloc(layers, sizeof(Matrix));
    Matrix* activations2 = calloc(layers, sizeof(Matrix));
    Matrix* preactivations1 = calloc(layersAdjust, sizeof(Matrix));
    Matrix* preactivations2 = calloc(layersAdjust, sizeof(Matrix));
    activations1[0] = initEmptyMatrix(1, size[0]);
    activations2[0] = initEmptyMatrix(1, size[0]);

    for (int i = 1; i < layers; i++) {
        wghts1[i - 1] = initNormal(size[i - 1], size[i], 0.0, deviation);
        wghts2[i - 1] = initNormal(size[i - 1], size[i], 0.0, deviation);
        biases1[i - 1] = initEmptyMatrix(1, size[i]);
        biases2[i - 1] = initEmptyMatrix(1, size[i]);
        activations1[i] = initEmptyMatrix(1, size[i]);
        activations2[i] = initEmptyMatrix(1, size[i]);
        preactivations1[i - 1] = initEmptyMatrix(1, size[i]);
        preactivations2[i - 1] = initEmptyMatrix(1, size[i]);
    }

	return (Network) {layers, size, wghts1, wghts2, biases1, biases2, activations1, activations2, preactivations1, preactivations2, (AdamwState) {0}};
}

Network initEmptyNet(int layers, int* size) {
	int* s = malloc(sizeof(int) * layers);
	s[0] = size[0];
	int layersAdjust = layers - 1;
	Matrix* wghts1 = malloc(layersAdjust * sizeof(Matrix)); Matrix* wghts2 = malloc(layersAdjust * sizeof(Matrix));
	Matrix* biases1 = malloc(layersAdjust * sizeof(Matrix)); Matrix* biases2 = malloc(layersAdjust * sizeof(Matrix));
	Matrix* activations1 = malloc(layers * sizeof(Matrix)); Matrix* activations2 = malloc(layers * sizeof(Matrix));
	Matrix* preactivations1 = malloc(layersAdjust * sizeof(Matrix)); Matrix* preactivations2 = malloc(layersAdjust * sizeof(Matrix));
	activations1[0] = initEmptyMatrix(1, size[0]); activations2[0] = initEmptyMatrix(1, size[0]);
	for (int i = 1; i < layers; i++) {
		wghts1[i-1] = initEmptyMatrix(size[i-1], size[i]); wghts2[i-1] = initEmptyMatrix(size[i-1], size[i]);
		biases1[i-1] = initEmptyMatrix(1, size[i]); biases2[i-1] = initEmptyMatrix(1, size[i]);
		activations1[i] = initEmptyMatrix(1, size[i]); activations2[i] = initEmptyMatrix(1, size[i]);
		preactivations1[i-1] = initEmptyMatrix(1, size[i]); preactivations2[i-1] = initEmptyMatrix(1, size[i]);
		s[i] = size[i];
	} return (Network) {layers, s, wghts1, wghts2, biases1, biases2, activations1, activations2, preactivations1, preactivations2, (AdamwState) {0}};
}
Matrix forward(Network net, Matrix input) {
	memcpy(net.activations1[0].d, input.d, input.n * sizeof(float));
	Matrix current = copyMatrix(input);
	for (int i = 0; i < net.layers-1; i++) {
		Matrix product = multiply(net.weights1[i], current);
		Matrix sum = addMatrix(net.biases1[i], product);
		//net.preactivations[i] = copyMatrix(sum);
		memcpy(net.preactivations1[i].d, sum.d, sum.n * sizeof(float));
		for (int j = 0; j < sum.n; j++) {
			sum.d[j] = sigmoid(sum.d[j]);
		}
		//net.activations[i+1] = copyMatrix(sum);
		memcpy(net.activations1[i+1].d, sum.d, sum.n * sizeof(float));
		free(product.d); free(current.d); current = sum;
	} return current;
}

void backward(AdamwState* adamw, Network* net, Matrix output, Matrix expected) {
    //Matrix delta = sccDerivitive(output, expected);
    Matrix delta = mseDerivative(output, expected);
    //AdamwState adamw = adamwInit(net, 0.01, 0.1);

    for (int i = 0; i < delta.n; i++ ) {
        delta.d[i] *= sigmoidDx(net->preactivations1[net->layers - 2].d[i]);
    }

    Matrix weightGrad[net->layers - 1], biasGrad[net->layers - 1];

    for (int i = net->layers - 2; i >= 0; i--) {
        Matrix aT = transpose(net->activations1[i]);
        weightGrad[i] = multiply(delta, aT);
        free(aT.d);
        biasGrad[i] = delta;
        if (i != 0) {
            Matrix wT = transpose(net->weights1[i]);
            delta = multiply(wT, delta);
            free(wT.d);
            for (int j = 0; j < delta.n; j++) {
                delta.d[j] *= sigmoidDx(net->preactivations1[i - 1].d[j]);
            }
        }
    }

    adamwOptimize(adamw, net, weightGrad, biasGrad);

    for (int i = 0; i < net->layers - 1; i++) {
        free(weightGrad[i].d);
        free(biasGrad[i].d);
    }
}

void train(Network* net, Matrix inputs[], Matrix expected[], int length, int epochs) {
    AdamwState adamw = adamwInit(*net, 0.01, 0.1);
	Matrix prediction;
	for (int i = 0; i < epochs; i++) {
	    float loss = 0.0;
		for (int j = 0; j < length; j++) {
			prediction = forward(*net, inputs[j]);
			loss += mseFunction(prediction, expected[j]);
			backward(&adamw, net, prediction, expected[j]);
			free(prediction.d);
		}

		if (i % 10 == 0 || i == epochs - 1) {
			printf("epoch %d  loss %f\n", i, loss / length);
		}
	}

	adamwFree(&adamw, *net);
}

Matrix predict(Network net, Matrix input) {
	Matrix current = copyMatrix(input);
	for (int i = 0; i < net.layers-1; i++) {
		Matrix product = multiply(net.weights1[i], current);
		Matrix sum = addMatrix(net.biases1[i], product);
		for (int j = 0; j < sum.n; j++) {sum.d[j] = sigmoid(sum.d[j]);}
		free(product.d); free(current.d); current = sum;
	} return current;
}

void freeNet(Network* net) {
	freeMatrixArr(net -> weights1, net -> layers - 1);
	freeMatrixArr(net -> biases1, net -> layers - 1);
	freeMatrixArr(net -> activations1, net -> layers);
	freeMatrixArr(net -> preactivations1, net -> layers - 1);
}

void printNet(Network net) {
    printf("Network:\n");

    printf("Layers: %d\n\n", net.layers);

    printf("Size:\n");
    for (int i = 0; i < net.layers; i++) {
        printf("%d, ", net.size[i]);
    }
    printf("\n\n");

    printf("Weights:\n");
    for (int i = 0; i < net.layers - 1; i++) {
        printf("%d:\n", i);
        print(net.weights1[i]);
    }
    printf("\n");

    printf("Biases:\n");
    for (int i = 0; i < net.layers - 1; i++) {
        printf("%d:\n", i);
        print(net.biases1[i]);
    }
    printf("\n");

    printf("Activations:\n");
    for (int i = 0; i < net.layers; i++) {
        printf("%d:\n", i);
        print(net.activations1[i]);
    }
    printf("\n");

    printf("Preactivations:\n");
    for (int i = 0; i < net.layers - 1; i++) {
        printf("%d:\n", i);
        print(net.preactivations1[i]);
    }
    printf("\n");
}

unsigned char* serializeNet(Network n) {
    int size = sizeof(int) + n.layers * sizeof(int);

    for (int i = 0; i < n.layers - 1; i++) {
        size += sizeofmatrix(n.weights1[i]);
        size += sizeofmatrix(n.biases1[i]);
        size += sizeofmatrix(n.activations1[i]);
        size += sizeofmatrix(n.preactivations1[i]);
    }

    size += sizeofmatrix(n.activations1[n.layers - 1]);

    unsigned char* bytes = malloc(size);
    unsigned char* p = bytes;
    unsigned char* tmp;

    memcpy(p, &n.layers, sizeof(int));
    p += sizeof(int);
    memcpy(p, n.size, n.layers * sizeof(int));
    p += n.layers * sizeof(int);

    for (int i = 0; i < n.layers - 1; i++) {
        tmp = serializeMatrix(n.weights1[i]);
        memcpy(p, tmp, sizeofmatrix(n.weights1[i]));
        free(tmp);
        p += sizeofmatrix(n.weights1[i]);

        tmp = serializeMatrix(n.biases1[i]);
        memcpy(p, tmp, sizeofmatrix(n.biases1[i]));
        free(tmp);
        p += sizeofmatrix(n.biases1[i]);

        tmp = serializeMatrix(n.activations1[i]);
        memcpy(p, tmp, sizeofmatrix(n.activations1[i]));
        free(tmp);
        p += sizeofmatrix(n.activations1[i]);

        tmp = serializeMatrix(n.preactivations1[i]);
        memcpy(p, tmp, sizeofmatrix(n.preactivations1[i]));
        free(tmp);
        p += sizeofmatrix(n.preactivations1[i]);
    }

    tmp = serializeMatrix(n.activations1[n.layers - 1]);
    memcpy(p, tmp, sizeofmatrix(n.activations1[n.layers - 1]));
    free(tmp);

    return bytes;
}

Network deserializeNet(unsigned char* m) {
    Network n = {0};
    unsigned char* p = m;

    memcpy(&n.layers, p, sizeof(int));
    p += sizeof(int);
    int layersAdjusted = n.layers - 1;

    n.size = malloc(n.layers * sizeof(int));
    memcpy(n.size, p, n.layers * sizeof(int));
    p += n.layers * sizeof(int);

    n.weights1 = malloc(layersAdjusted * sizeof(Matrix));
    n.biases1 = malloc(layersAdjusted * sizeof(Matrix));
    n.activations1 = malloc(n.layers * sizeof(Matrix));
    n.preactivations1 = malloc(layersAdjusted * sizeof(Matrix));

    for (int i = 0; i < layersAdjusted; i++) {
        n.weights1[i] = deserializeMatrix(p);
        p += sizeofmatrix(n.weights1[i]);

        n.biases1[i] = deserializeMatrix(p);
        p += sizeofmatrix(n.biases1[i]);

        n.activations1[i] = deserializeMatrix(p);
        p += sizeofmatrix(n.activations1[i]);

        n.preactivations1[i] = deserializeMatrix(p);
        p += sizeofmatrix(n.preactivations1[i]);
    }

    n.activations1[layersAdjusted] = deserializeMatrix(p);

	return n;
}
