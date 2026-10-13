#include <stdio.h>

typedef struct {
	int w, h, n;
	float* d;
} Matrix;

Matrix multiply(Matrix m1, Matrix m2);

Matrix elementwise(Matrix m1, Matrix m2);

Matrix transpose(Matrix m);

Matrix scale(Matrix m, float s);

Matrix add(Matrix m, float a);

unsigned char* serialize(Matrix m, FILE* f);

Matrix deserialize(unsigned char* m);
