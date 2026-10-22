// ALL matrices below should be treated as vectors
#include "../matrix/matrix.h"
#include "feedforward_old.h"
#include "adamw.h"
#include "normal.h"
#include "costs.h"
#include "activations.h"

Old_Network old_init(int layers, int* size, float deviation) {
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

	return (Old_Network) {layers, size, wghts, biases, activations, preactivations};
}

Old_Network old_initEmptyNet(int layers, int* size) {
	int* s = malloc(sizeof(int) * layers);
	s[0] = size[0];
	int layersAdjust = layers - 1; Matrix* wghts = malloc(layersAdjust * sizeof(Matrix));
	Matrix* biases = malloc(layersAdjust * sizeof(Matrix)); Matrix* activations = malloc(layers * sizeof(Matrix));
	Matrix* preactivations = malloc(layersAdjust * sizeof(Matrix)); activations[0] = initEmptyMatrix(1, size[0]);
	for (int i = 1; i < layers; i++) {
		wghts[i-1] = initEmptyMatrix(size[i-1], size[i]); biases[i-1] = initEmptyMatrix(1, size[i]);
		activations[i] = initEmptyMatrix(1, size[i]); preactivations[i-1] = initEmptyMatrix(1, size[i]);
		s[i] = size[i];
	} return (Old_Network) {layers, s, wghts, biases, activations, preactivations};
}
Matrix old_forward(Old_Network net, Matrix input) {
	memcpy(net.activations[0].d, input.d, input.n * sizeof(float));
	Matrix current = copyMatrix(input);
	for (int i = 0; i < net.layers-1; i++) {
		Matrix product = multiply(net.weights[i], current);
		Matrix sum = addMatrix(net.biases[i], product);
		memcpy(net.preactivations[i].d, sum.d, sum.n * sizeof(float));
		for (int j = 0; j < sum.n; j++) {
			sum.d[j] = sigmoid(sum.d[j]);
		}
		memcpy(net.activations[i+1].d, sum.d, sum.n * sizeof(float));
		free(product.d); free(current.d); current = sum;
	} return current;
}

void old_backward(AdamwState* adamw, Old_Network* net, Matrix output, Matrix expected) {
    Matrix delta = mseDerivative(output, expected);

    for (int i = 0; i < delta.n; i++ ) {
        delta.d[i] *= sigmoidDx(net->preactivations[net->layers - 2].d[i]);
    }

    Matrix weightGrad[net->layers - 1], biasGrad[net->layers - 1];

    for (int i = net->layers - 2; i >= 0; i--) {
        Matrix aT = transpose(net->activations[i]);
        weightGrad[i] = multiply(delta, aT);
        free(aT.d);
        biasGrad[i] = delta;
        if (i != 0) {
            Matrix wT = transpose(net->weights[i]);
            delta = multiply(wT, delta);
            free(wT.d);
            for (int j = 0; j < delta.n; j++) {
                delta.d[j] *= sigmoidDx(net->preactivations[i - 1].d[j]);
            }
        }
    }

//    adamwOptimize(adamw, net, weightGrad, biasGrad);

    for (int i = 0; i < net->layers - 1; i++) {
        free(weightGrad[i].d);
        free(biasGrad[i].d);
    }
}

void old_train(Old_Network* net, Matrix inputs[], Matrix expected[], int length, int epochs) {
//    AdamwState adamw = adamwInit(*net, 0.01, 0.1);
	Matrix prediction;
	for (int i = 0; i < epochs; i++) {
	    float loss = 0.0;
		for (int j = 0; j < length; j++) {
//			old_prediction = forward(*net, inputs[j]);
			loss += mseFunction(prediction, expected[j]);
//			old_backward(&adamw, net, prediction, expected[j]);
			free(prediction.d);
		}

		if (i % 10 == 0 || i == epochs - 1) {
			printf("epoch %d  loss %f\n", i, loss / length);
		}
	}

//	adamwFree(&adamw, *net);
}

Matrix old_predict(Old_Network net, Matrix input) {
	Matrix current = copyMatrix(input);
	for (int i = 0; i < net.layers-1; i++) {
		Matrix product = multiply(net.weights[i], current);
		Matrix sum = addMatrix(net.biases[i], product);
		for (int j = 0; j < sum.n; j++) {sum.d[j] = sigmoid(sum.d[j]);}
		free(product.d); free(current.d); current = sum;
	} return current;
}

void old_freeNet(Old_Network* net) {
	freeMatrixArr(net -> weights, net -> layers - 1);
	freeMatrixArr(net -> biases, net -> layers - 1);
	freeMatrixArr(net -> activations, net -> layers);
	freeMatrixArr(net -> preactivations, net -> layers - 1);
}

void old_printNet(Old_Network net) {
    printf("Old_Network:\n");

    printf("Layers: %d\n\n", net.layers);

    printf("Size:\n");
    for (int i = 0; i < net.layers; i++) {
        printf("%d, ", net.size[i]);
    }
    printf("\n\n");

    printf("Weights:\n");
    for (int i = 0; i < net.layers - 1; i++) {
        printf("%d:\n", i);
        print(net.weights[i]);
    }
    printf("\n");

    printf("Biases:\n");
    for (int i = 0; i < net.layers - 1; i++) {
        printf("%d:\n", i);
        print(net.biases[i]);
    }
    printf("\n");

    printf("Activations:\n");
    for (int i = 0; i < net.layers; i++) {
        printf("%d:\n", i);
        print(net.activations[i]);
    }
    printf("\n");

    printf("Preactivations:\n");
    for (int i = 0; i < net.layers - 1; i++) {
        printf("%d:\n", i);
        print(net.preactivations[i]);
    }
    printf("\n");
}

unsigned char* old_serializeOld_Network(Old_Network n) {
    int size = sizeof(int) + n.layers * sizeof(int);

    for (int i = 0; i < n.layers - 1; i++) {
        size += sizeofmatrix(n.weights[i]);
        size += sizeofmatrix(n.biases[i]);
        size += sizeofmatrix(n.activations[i]);
        size += sizeofmatrix(n.preactivations[i]);
    }

    size += sizeofmatrix(n.activations[n.layers - 1]);

    unsigned char* bytes = malloc(size);
    unsigned char* p = bytes;
    unsigned char* tmp;

    memcpy(p, &n.layers, sizeof(int));
    p += sizeof(int);
    memcpy(p, n.size, n.layers * sizeof(int));
    p += n.layers * sizeof(int);

    for (int i = 0; i < n.layers - 1; i++) {
        tmp = serializeMatrix(n.weights[i]);
        memcpy(p, tmp, sizeofmatrix(n.weights[i]));
        free(tmp);
        p += sizeofmatrix(n.weights[i]);

        tmp = serializeMatrix(n.biases[i]);
        memcpy(p, tmp, sizeofmatrix(n.biases[i]));
        free(tmp);
        p += sizeofmatrix(n.biases[i]);

        tmp = serializeMatrix(n.activations[i]);
        memcpy(p, tmp, sizeofmatrix(n.activations[i]));
        free(tmp);
        p += sizeofmatrix(n.activations[i]);

        tmp = serializeMatrix(n.preactivations[i]);
        memcpy(p, tmp, sizeofmatrix(n.preactivations[i]));
        free(tmp);
        p += sizeofmatrix(n.preactivations[i]);
    }

    tmp = serializeMatrix(n.activations[n.layers - 1]);
    memcpy(p, tmp, sizeofmatrix(n.activations[n.layers - 1]));
    free(tmp);

    return bytes;
}

Old_Network old_deserializeOld_Network(unsigned char* m) {
    Old_Network n = {0};
    unsigned char* p = m;

    memcpy(&n.layers, p, sizeof(int));
    p += sizeof(int);
    int layersAdjusted = n.layers - 1;

    n.size = malloc(n.layers * sizeof(int));
    memcpy(n.size, p, n.layers * sizeof(int));
    p += n.layers * sizeof(int);

    n.weights = malloc(layersAdjusted * sizeof(Matrix));
    n.biases = malloc(layersAdjusted * sizeof(Matrix));
    n.activations = malloc(n.layers * sizeof(Matrix));
    n.preactivations = malloc(layersAdjusted * sizeof(Matrix));

    for (int i = 0; i < layersAdjusted; i++) {
        n.weights[i] = deserializeMatrix(p);
        p += sizeofmatrix(n.weights[i]);

        n.biases[i] = deserializeMatrix(p);
        p += sizeofmatrix(n.biases[i]);

        n.activations[i] = deserializeMatrix(p);
        p += sizeofmatrix(n.activations[i]);

        n.preactivations[i] = deserializeMatrix(p);
        p += sizeofmatrix(n.preactivations[i]);
    }

    n.activations[layersAdjusted] = deserializeMatrix(p);

	return n;
}
