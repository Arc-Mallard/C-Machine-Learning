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

    if (res.d == NULL) {
        perror("calloc");
        return res;
    }

    for (int i = 0; i < m1.h; i++) {
        for (int j = 0; j < m2.w; j++) {
            for (int k = 0; k < m1.w; k++) {
                res.d[i * res.w + j] += m1.d[i * m1.w + k] * m2.d[k * m2.w + j];
            }
        }
    }

	return res;
}

Matrix elementwise(Matrix m1, Matrix m2) {
	//if (!m1.d || !m2.d || m1.n != m2.n) {perror("elementwise"); return m1;}
	Matrix e = {m1.w, m1.h, m1.n, malloc(m1.n * sizeof(float))};
	for (int i = 0; i < m1.n; i++) {e.d[i] = m1.d[i] * m2.d[i];}
	return e;
}

Matrix transpose(Matrix m) {
	//if (!m.d) {perror("transpose"); return m;}
	Matrix t = {
		m.h, m.w, m.n,
		malloc(m.n * sizeof(float))
	};
	for (int i = 0;  i < m.h; i++) {
		for (int j = 0; j < m.w; j++) {
			t.d[j * t.w + i] = m.d[i * m.w + j];
		}
	} return t;
}

Matrix scale(Matrix m, float s) {
    Matrix res = {0};
    res.h = m.h;
    res.w = m.w;
    res.n = res.h * res.w;
    res.d = malloc(res.n * sizeof(float));

    if (res.d == NULL) {
        perror("malloc");
        return res;
    }

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

    if (res.d == NULL) {
        perror("malloc");
        return res;
    }

    for (int i = 0; i < res.n; i++) {
        res.d[i] = m.d[i] + a;
    }

    return res;
}

void print(Matrix m) {
    for (int i = 0; i < m.h; i++) {
        printf("[ ");
        for (int j = 0; j < m.w; j++) {
            printf("%f ", m.d[i * m.w + j]);
        }
        printf("]\n");
    }
}

unsigned char* serialize(Matrix m) {
	unsigned char* bytes = malloc(3 * sizeof(int) + m.n * sizeof(float));
	//if (!bytes || !m.d) {perror("serialize"); return bytes;}
	memcpy(bytes, &m.w, sizeof(int));
	memcpy(bytes + 1 * sizeof(int), &m.h, sizeof(int));
	memcpy(bytes + 2 * sizeof(int), &m.n, sizeof(int));
	memcpy(bytes + 3 * sizeof(int), m.d, m.n * sizeof(float));
	return bytes;
}

Matrix deserialize(unsigned char* m) {
	Matrix d = {0};
	memcpy(&d.w, m, sizeof(int));
	memcpy(&d.h, m + 1 * sizeof(int), sizeof(int));
	memcpy(&d.n, m + 2 * sizeof(int), sizeof(int));
	d.d = malloc(d.n * sizeof(float));
	//if (!d.d) {perror("deserialize"); return d;}
	memcpy(d.d, m + 3 * sizeof(int), d.n * sizeof(float));
	return d;
}
