#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPSILON 1e-7f

typedef struct {
	int w, h, n;
	float* d;
} Matrix;

typedef struct {
	int layers; int* size; Matrix* weights;
	// These matrices below are actually all vectors
	Matrix* biases; Matrix* activations; Matrix* preactivations;
} Network;

Matrix initEmptyMatrix(int w, int h);

Network initEmptyNet(int layers, int* size);

void freeMatrixArr(Matrix* m, int len);

Matrix multiply(Matrix m1, Matrix m2);

Matrix elementwise(Matrix m1, Matrix m2);

Matrix transpose(Matrix m);

Matrix scale(Matrix m, float s);

Matrix add(Matrix m, float a);

Matrix addMatrix(Matrix m1, Matrix m2);

Matrix copyMatrix(Matrix m);

void print(Matrix m);

unsigned char* serializeMatrix(Matrix m);

Matrix deserializeMatrix(unsigned char* m);

#endif
