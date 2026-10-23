/*
 * main.c
 *
 *  Created on: Oct 9, 2026
 *      Author: nicol
 */

#include "old/feedforward.h"
#include "neural/normal.h"
#include <time.h>

int main() {
    srand(time(NULL));

	float data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	Matrix test1 = {2, 3, 6, data1};
	printf("declaring matrix 1 \n");
	print(test1); printf("\n");

	float data2[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
	Matrix test2 = {3, 2, 6, data2};
	printf("declaring matrix 2 \n");
	print(test2); printf("\n");

	Matrix test3 = multiply(test1, test2);
	printf("multiplying matrix 1 with matrix 2 \n");
	print(test3); printf("\n");

	Matrix test4 = transpose(test3);
	printf("transposing multiplied matrix \n");
	print(test4); printf("\n");

	printf("serializing transposed matrix \n");
	unsigned char* bytes = serializeMatrix(test4);
	print(deserializeMatrix(bytes));
	free(bytes);
	printf("\n");

	printf("gaussian noise\n");
	int low = 0, high = 0, mid = 0;
	for (int i = 0; i < 1000; i++) {
	    float value = gaussianNoise(0, .5);
		if (value > 0.5)
		    high++;
		else if (value < -0.5)
		    low++;
		else
		    mid++;
		//printf("%f\n", value);
	}
	printf("low: %d, mid: %d, high: %d\n\n", low, mid, high);

	printf("initialize test matrix\n");
	Matrix test5 = initNormal(3, 3, 0.0, 1.0);
	print(test5);
	printf("\n");

	printf("initialize neural network");
	int sizes[] = {2, 3, 3, 1};
	Network net = init(4, sizes, 0.1f);
	printNet(net); printf("\n");

	printf("test forward propagation\n");
	float vector[] = {1.0, 2.0};
	Matrix input = {1, 2, 2, vector};
	Matrix output = forward(net, input);
	print(output); printf("\n");

	printf("prediction: \n");
	Matrix target = predict(net, input);
	print(target); printf("\n");

	/*
	printf("back propagation test\n");
	Network backtest = backward(net, output, target);
	printNet(backtest); printf("\n");
	*/

	printf("training\n");
	Matrix* inputs = malloc(sizeof(Matrix)*500);
	Matrix* targets = malloc(sizeof(Matrix)*500);
	for (int i = 0; i < 500; i++) {
		inputs[i] = initEmptyMatrix(1, 2);
		targets[i] = initEmptyMatrix(1, 1);
		for (int j = 0; j < inputs[i].n; j++) {
			inputs[i].d[j] = (((float)rand() + 1.0) / ((float)RAND_MAX + 2.0) / 2);
		} targets[i].d[0] = inputs[i].d[0] + inputs[i].d[1];
	}

	train(&net, inputs, targets, 500, 100);

	printNet(net);
	printf("predicting: \n");
	float testValue1[] = {0.2, 0.5};
	Matrix inputValue1 = {1, 2, 2, testValue1};
	target = predict(net, inputValue1);
	printf("0.2 + 0.5 = "); print(target); printf("\n");

	float testValue2[] = {0.5, 0.4};
	Matrix inputValue2 = {1, 2, 2, testValue2};
	target = predict(net, inputValue2);
	printf("0.5 + 0.4 = "); print(target); printf("\n");

	float testValue3[] = {0.1, 0.1};
	Matrix inputValue3 = {1, 2, 2, testValue3};
	target = predict(net, inputValue3);
	printf("0.1 + 0.1 = "); print(target); printf("\n");

	printf("serialize network\n");
	bytes = serializeNet(net);
	printNet(deserializeNet(bytes));
	free(bytes);
	printf("\n");

	printf("Hello World!\n");
}
