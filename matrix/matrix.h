#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	int w, h, n;
	float* d;
} Matrix;

typedef struct {
	int layers; int* size; Matrix* weights;
	// These matrices below are actually all vectors
	Matrix* biases; Matrix* activations; Matrix* preactivations;
} Network;

Matrix multiply(Matrix m1, Matrix m2);

Matrix elementwise(Matrix m1, Matrix m2);

Matrix transpose(Matrix m);

Matrix scale(Matrix m, float s);

Matrix add(Matrix m, float a);

void print(Matrix m);

unsigned char* serializeMatrix(Matrix m);

Matrix deserializeMatrix(unsigned char* m);
