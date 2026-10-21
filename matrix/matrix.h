#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../types.h"

#define EPSILON 1e-7f
#define sizeofmatrix(m) (3 * sizeof(int) + m.n * sizeof(float))

Matrix initEmptyMatrix(int w, int h);

void freeMatrixArr(Matrix* m, int len);

void freeMatrix(Matrix* m);

Matrix multiply(Matrix m1, Matrix m2);

Matrix elementwise(Matrix m1, Matrix m2);

Matrix transpose(Matrix m);

Matrix scale(Matrix m, float s);

Matrix add(Matrix m, float a);

Matrix addMatrix(Matrix m1, Matrix m2);

Matrix subtractMatrix(Matrix m1, Matrix m2);

Matrix subtractMatrixFrom(Matrix m1, float n);

Matrix copyMatrix(Matrix m);

Matrix* copyMatrixArr(Matrix* m, int len);

void print(Matrix m);

unsigned char* serializeMatrix(Matrix m);

Matrix deserializeMatrix(unsigned char* m);

#endif
