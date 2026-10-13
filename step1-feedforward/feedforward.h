typedef struct {
	
} Network;

// ALL matricies below should be treated as vectors 

Network train(Matrix inputs[], Matrix expected[], int epochs);

Matrix predict(Matrix input);

unsigned char* serialize(Network n);

Network deserialize(unsigned char* m);