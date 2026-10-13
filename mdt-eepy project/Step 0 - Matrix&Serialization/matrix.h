
typedef struct {
	int w, h;
	float* d[];
} Matrix;

Matrix multiply(Matrix m1, Matrix m2);

Matrix elementwise(Matrix m1, Matrix m2);

Matrix transpose(Matrix m);

Matrix scale(Matrix m, float s);

Matrix add(Matrix m, float a);

char* serialize(Matrix m);

Matrix deserialize(char* m);
