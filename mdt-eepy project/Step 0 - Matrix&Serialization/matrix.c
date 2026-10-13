#include "matrix.h"

Matrix multiply(Matrix m1, Matrix m2) {
    Matrix res = {0};

    if (m1.w != m2.h) {
        return res;
    }

    res.h = m1.h;
    res.w = m2.w;
    res.n = res.h * res.w;
    res.d = calloc(res.n, sizeof(float));

    for (int i = 0; i < m1.h; i++) {
        for (int j = 0; j < m2.w; j++) {
            for (int k = 0; m1.w; k++) {
                res.d[i * res.w + j] += m1.d[i * m1.w + k] * m2.d[k * m2.w + j];
            }
        }
    }

	return res;
}

Matrix elementwise(Matrix m1, Matrix m2) {
	if (!m1 || !m2 || m1 -> n != m2 -> n) {perror("elementwise");}
	Matrix e = {m1 -> w, m1 -> h, m1 -> n, malloc(m1 -> n * sizeof(float))};
	for (int i = 0; i < m1 -> n; i++) {e -> d[i] = m1 -> d[i] * m2 -> d[i];}
	return e;
}

Matrix transpose(Matrix m) {
	if (!m) {perror("transpose");}
	Matrix t = {
		m -> w, m -> h, m -> n,
		malloc(m -> n * sizeof(float))
	};
	for (int i = 0;  i < m -> w; i++) {
		for (int j = 0; j < m -> h; j++) {


//			[0, 0, 1, 1, 2, 2]

//			|0, 0|
//			|1, 1|  ->	|0  1, 2|
//			|2, 2|		|0, 1, 2|

		}
	}
	return NULL;
}

Matrix scale(Matrix m, float s) {
    Matrix res = {0};
    res.h = m.h;
    res.w = m.w;
    res.n = res.h * res.w;
    res.d = malloc(res.n * sizeof(float));

    for (int i = 0; i < res.n; i++) {
        res.d[i] = m.d[i] * s;
    }

	return res;
}

Matrix add(Matrix m, float a) {
    Matrix res = {0};
    res.h = m.h;
    res.w = m.w;
    res.n = res.h * res.w;
    res.d = malloc(res.n * sizeof(float));

    for (int i = 0; i < res.n; i++) {
        res.d[i] = m.d[i] + a;
    }

    return res;
}

void print(Matrix m) {

}

unsigned char* serialize(Matrix m, FILE* f) {
	return NULL;
}

Matrix deserialize(unsigned char* m) {
	return NULL;
}
