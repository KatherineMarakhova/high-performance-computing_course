#include <omp.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
// #include <stdio.h>
using namespace std;


void printm(double ** m, int n, string filename){
    std::ofstream out;                  // поток для записи
    out.open(filename);                 // открываем файл для записи
    if (out.is_open())
    {
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                out << m[i][j] << " ";
            }
            out << endl;
        }
    }
    else{
        cout << "Не удалось открыть файл \n";
    }
    out.close();                      // закрываем файл
}

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

double** set_values(int n, int max, int min) {
    double** matrix = new double*[n];
    for (int i = 0; i < n; ++i) {
        matrix[i] = new double[n];
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = (double)(rand())/RAND_MAX*(max - min) + min;
        }
    }
    return matrix;
}

double ** matmul(double** a, double** b, int n) {
    
    double** res = init_matrix(n);                          //общая переменная, используется всеми потоками
    int i, j, k;
    
    #pragma omp parallel for private(i,j,k) shared(a,b,res)
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
    int i, j;
    #pragma omp parallel for private(i,j) shared(a,b,c,result)
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            result[i][j] = a[i][j]+b[i][j]+c[i][j];
        }
    }
    return result;
}

double calculate(double** B, double** C, double** E, double** D,int n){
    double tim = omp_get_wtime();
    double** B3 = matcube(B, n);
    double** C2E = matsquare_and_mul(C, E, n);
    double** BD = matmul(B, D, n);
    double** result = sum_all(B3, C2E, BD, n);
    tim = omp_get_wtime()-tim;
    printm(result, n, "result.txt");
    return tim;
}

int main(){
    int n = 100;
    // int numt = 10;
    
    //диапазон рандомных чисел
    int fMin = 1; 
    int fMax = 10; 
    
    //заполняем матрицы
    double** B = set_values(n, fMax, fMin);
    printm(B, n, "B.txt");
    double** C = set_values(n, fMax, fMin);
    printm(C, n, "C.txt");
    double** D = set_values(n, fMax, fMin);
    printm(D, n, "D.txt");
    double** E = ones(n);
    printm(E, n, "E.txt");

    for (int thr=1; thr<11; thr++){
        for(int i=0; i<3; i++){
            double cur_result;
            omp_set_num_threads(thr);
            cur_result = calculate(B, C, E, D, n);
            printf("Threads: %d; Elapsed time: %.4f\n", thr, cur_result);
        }
    }
    

    return 0;
}
