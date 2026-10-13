/*
 * main.c
 *
 *  Created on: Oct 9, 2026
 *      Author: nicol
 */

#include "matrix/matrix.h"

int main() {

	float* data1 = {1, 2, 3, 4, 5, 6};
	Matrix test1 = {2, 3, 6, data1};
	print(test1);

	float* data2 = {7, 8, 9, 10, 11, 12};
	Matrix test2 = {2, 3, 6, data2};
	print(test2);

	Matrix test3 = multiply();

	printf("Hello World!");
}
