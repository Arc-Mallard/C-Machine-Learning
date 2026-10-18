/*
 * softmax.c
 *
 *  Created on: Oct 18, 2026
 *      Author: nicol
 */

#include "activation.h"

Matrix actFunction(Matrix m) {
	float ttl = max(m), sum = 0;
	Matrix a = initEmptyMatrix(m.w, m.h);
	for (int i = 0; i < m.n; i++) {
		a.d[i] = expf(m.d[i] - ttl); sum += a.d[i];
	} for (int i = 0; i < m.n; i++) {a.d[i] /= sum;}
	return a;
}

float max(Matrix m) {
	float max = m.d[0];
	for (int i = 1; i < m.n; i++) {
		max = m.d[i] > max? m.d[i] : max;
	} return max;
}
