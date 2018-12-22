#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "time.h"
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void printMatr(float *M, int rows, int cols)
{
	if (rows*cols < 17) {
		printf("Matr:\n");
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				printf("%6.2f\t", M[i*cols + j]);
			}
			printf("\n");
		}
	}
}
void matrMult(float* A, float* B, float* C, int rowsA, int colsA, int colsB)
{
	for (int i = 0; i < rowsA; ++i) {
		for (int j = 0; j < colsB; ++j) {
			for (int k = 0; k < colsA; ++k) {
				C[i*colsB + j] += A[i*colsA + k] * B[k*colsB + j];
			}
		}
	}
}
void generate(float* &A, float* &b, int size) {
	srand(time(NULL));
	printf("start generate\n");
	for (int j = 0; j < size; ++j) {
		for (int i = 0; i < size; ++i) {
			//A[i*size + j] = rand() % 100;
			//A[IDX2C(i, j, size)] = i * size + j;
			A[IDX2C(i, j, size)] = rand() % 10;
		}
		b[j] = rand() % 100;
	}
	float* transpA = new float[size];
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			transpA[i*size + j] = A[IDX2C(i, j, size)];
		}
	}
	float* temp = new float[size*size];
	printMatr(A, size, size);
	printMatr(transpA, size, size);
	matrMult(A, transpA, temp, size, size, size);
	//multiplicating matrix on transponed itself generates a
	//positive SEMIdefinite (>= 0)
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			A[IDX2C(i, j, size)] = temp[IDX2C(i, j, size)] / (float)100;
		}
	}
	printMatr(A, size, size);
	printMatr(b, size, 1);
}

int main(int argc, char** argv)
{
	cublasStatus_t stat;
	cublasHandle_t handle;
	//int size = atoi(argv[1]);
	//int iters = atoi(argv[2]);
	int size = 4;
	int iters = 1000;
	float *A = new float[size*size], *b = new float[size], *d_A, *d_b, *r, arr, arar, *x, *ar, *x0 = new float[size], t;
	generate(A, b, size);
	printf("finish gen\n");
	for (int i = 0; i < size; ++i) {
		x0[i] = 1;
	}
	printf("filled x0\n");
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		cublasDestroy(handle);
		printf("CUBLAS initialization failed\n");
		return EXIT_FAILURE;
	}
	printf("start performing malloc\n");
	cudaMalloc((void**)&d_A, size*size*sizeof(float));
	cudaMalloc((void**)&d_b, size * sizeof(float));
	cudaMalloc((void**)&r, size * sizeof(float));
	cudaMalloc((void**)&ar, size * sizeof(float));
	cudaMalloc((void**)&x, size * sizeof(float));
	printf("performed malloc\n");
	stat = cublasSetMatrix(size, size, sizeof(*A), A, size, d_A, size);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		cublasDestroy(handle);
		printf("cublas setmatrix error\n");
		return EXIT_FAILURE;
	}
	stat = cublasSetVector(size, sizeof(float), b, 1, d_b, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		cublasDestroy(handle);
		printf("cublas setvector error\n");
		return EXIT_FAILURE;
	}
	stat = cublasSetVector(size, sizeof(float), x0, 1, x, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		cublasDestroy(handle);
		printf("cublas setvector error\n");
		return EXIT_FAILURE;
	}
	float alpha = 1.0f;
	float beta = -1.0f;
	float zero = 0.0f;
	float eps = 0.001f;
	bool flag = false;
	printf("begin iters\n");
	int it;
	for (it = 0; it < iters; ++it) {
		stat = cublasScopy(handle, size, d_b, 1, r, 1); //from d_b into r
		if (stat != CUBLAS_STATUS_SUCCESS) {
			cublasDestroy(handle);
			printf("cublasSgemv  error\n");
			return EXIT_FAILURE;
		}
		stat = cublasSgemv(handle, CUBLAS_OP_N, size, size, &alpha, d_A, size, x, 1, &beta, r, 1);// r = Ax - b; ~ ;r = Ax - r
		if (stat != CUBLAS_STATUS_SUCCESS) {
			cublasDestroy(handle);
			printf("cublasSgemv  error\n");
			return EXIT_FAILURE;
		}
		stat = cublasSgemv(handle, CUBLAS_OP_N, size, size, &alpha, d_A, size, r, 1, &zero, ar, 1);//Ar into ar
		if (stat != CUBLAS_STATUS_SUCCESS) {
			cublasDestroy(handle);
			printf("cublasSgemv error\n");
			return EXIT_FAILURE;
		}
		stat = cublasSdot(handle, size, ar, 1, r, 1, &arr); //(Ar,r) 
		if (stat != CUBLAS_STATUS_SUCCESS) {
			cublasDestroy(handle);
			printf("cublasSdot error\n");
			return EXIT_FAILURE;
		}
		stat = cublasSdot(handle, size, ar, 1, ar, 1, &arar); //(Ar,Ar) 
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf("cublasSdot error\n"); 
			cublasDestroy(handle);
			return EXIT_FAILURE;
		} 
		if (arar == 0) { 
			printf("arar = 0\n");
			cublasDestroy(handle);
			return EXIT_FAILURE;
		}
		t = - (arr / (float)arar);
		stat = cublasSaxpy(handle, size, &t, r, 1, x, 1); //x = x - tr
		if (stat != CUBLAS_STATUS_SUCCESS) {
			cublasDestroy(handle);
			printf("cublasSaxpy error\n");
			return EXIT_FAILURE;
		}
		stat = cublasSgemv(handle, CUBLAS_OP_N, size, size, &alpha, d_A, size, x, 1, &zero, r, 1);//Ax into r
		if (stat != CUBLAS_STATUS_SUCCESS) {
			cublasDestroy(handle);
			printf("cublasSgemv error\n");
			return EXIT_FAILURE;
		}
		stat = cublasSaxpy(handle, size, &beta, d_b, 1 ,r, 1); //r = r - d_b
		if (stat != CUBLAS_STATUS_SUCCESS) {
			cublasDestroy(handle);
			printf("cublasSaxpy error\n");
			return EXIT_FAILURE;
		}
		stat = cublasGetVector(size, sizeof(float), r, 1, x0, 1);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			cublasDestroy(handle);
			printf("cublas get vector");
			cublasDestroy(handle);
			return EXIT_FAILURE;
		}
		flag = true;
		for (int i = 0; i < size; ++i) {
			if (abs(x0[i]) > eps) { 
				//printf("more than eps %d\n", i);
				flag = false;
			}
		}
		if (flag) break;
	}
	printf("iters: %d\n", it);
	stat = cublasGetVector(size, sizeof(float), x, 1, x0, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) {cublasDestroy(handle);
		printf("cublas get vector");
		cublasDestroy(handle);
		return EXIT_FAILURE;
	}
	cudaFree(d_A);
	cudaFree(d_b);
	cudaFree(x);
	cudaFree(ar);
	cudaFree(r);
	cublasDestroy(handle);
	printf("end iters\n");
	printMatr(x0, size, 1);
	delete[]A; delete[]x0; delete[]b;
	return EXIT_SUCCESS;
}
