/*
 * main.c
 *
 *  Created on: Oct 9, 2026
 *      Author: nicol
 */

#include "matrix/matrix.h"

int main() {

	float* data = {1, 2, 3, 4, 5, 6};
	Matrix test = {2, 3, 6, data};

	print(test);

	printf("Hello World!");
}
