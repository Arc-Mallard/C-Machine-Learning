/*
 * main.c
 *
 *  Created on: Oct 9, 2026
 *      Author: nicol
 */

#include "matrix/matrix.h"

int main() {

	float data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	Matrix test1 = {2, 3, 6, data1};
	print(test1);

	float data2[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
	Matrix test2 = {2, 3, 6, data2};
	print(test2);

	Matrix test3 = multiply(test1, test1);
	print(test3);

	print(transpose(test3));

	printf("Hello World!");
}
