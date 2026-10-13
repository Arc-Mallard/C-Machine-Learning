#include "matrix.h"

Matrix multiply(Matrix m1, Matrix m2) {
    printf("multiply");
	return NULL;
}

Matrix elementwise(Matrix m1, Matrix m2) {
	if (!m1 || !m2 || m1 -> n != m2 -> n) {return NULL;}
	Matrix e = {m1 -> w, m1 -> h, m1 -> n, malloc(m1 -> n * sizeof(float))};
	for (int i = 0; i < m1 -> n; i++) {e -> d[i] = m1 -> d[i] * m2 -> d[i];}
	return e;
}

Matrix transpose(Matrix m) {
	return NULL;
}

Matrix scale(Matrix m, float s) {
	return NULL;
}

Matrix add(Matrix m, float a) {
	return NULL;
}

unsigned char* serialize(Matrix m, FILE* f) {
	return NULL;
}

Matrix deserialize(unsigned char* m) {
	return NULL;
}
