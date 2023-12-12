#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
// #include <stdio.h>


double ** init_matrix(int n) {
    double ** matrix = new double*[n];
    for (int i = 0; i < n; i++) {
        matrix[i] = new double[n];
    }
    return matrix;
}

double ** ones(int n){
    double** ones = init_matrix(n);
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            ones[i][j]=1;
        }
    }
    return ones;
}

double ** set_values(double** a, double fMin, double fMax, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            double f = (double)rand() / RAND_MAX;
            a[i][j] = fMin + f * (fMax - fMin);
        }
    }
    return a;
}

double ** matmul(double** a, double** b, int n) {
    double** res = init_matrix(n);
    int i, j, k;
    // #pragma omp parallel for private(i,j,k) shared(a,b,res)
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            for (k = 0; k < n; ++k) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
} 

double ** matcube(double** a, int n){
    double ** sqr = matmul(a, a, n);
    return matmul(a, sqr, n);
}

double ** matsquare_and_mul(double** a, double** b, int n){
    double ** sqr = matmul(a, a, n);
    return matmul(sqr, b, n);
}

double ** sum_all(double** a, double** b, double** c, int n){
    double** result = init_matrix(n);
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            result[i][j] = a[i][j]+b[i][j]+c[i][j];
        }
    }
    return result;
}

void printm(double ** m, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            printf("%.3f ", m[i][j]);
        }
        printf("\n");
    }
}

int main(){
    int n = 100;
    
    //диапазон рандомных чисел
    int fMin = 1; 
    int fMax = 10; 
    
    //заполняем матрицы
    double** B = init_matrix(n);
    B = set_values(B, fMin, fMax, n);
    double** C = init_matrix(n);
    C = set_values(B, fMin, fMax, n);
    double** D = init_matrix(n);
    D = set_values(B, fMin, fMax, n);
    double** E = ones(n);

    //вычисления
    double tim = omp_get_wtime();

    double** B3 = matcube(B, n);
    double** C2E = matsquare_and_mul(C, E, n);
    double** BD = matmul(B, D, n);
    double** result = sum_all(B3, C2E, BD, n);

    // printf("\nResult: \n");
    // printm(result, n);
    tim = omp_get_wtime()-tim;
    printf("Time simple result: %.6f\n", tim);
    
    return 0;
}
