/*
 * main.c
 *
 *  Created on: Oct 9, 2026
 *      Author: nicol
 */

#include "matrix/matrix.h"

int main() {

	float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	Matrix test = {2, 3, 6, data};

	print(test);

	printf("Hello World!");
}
