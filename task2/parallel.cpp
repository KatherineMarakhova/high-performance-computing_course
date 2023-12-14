#include <iostream>
#include <fstream>
#include <omp.h>
#include <mpi.h>
#include <vector>

using namespace std;

const int mmax = 10;
const int mmin = 0;
int N = 50;
int size; 
int proc_rank;

// Функция для умножения двух матриц
double* matmul(double*& A, double*& B, int n) {
    double* tempm{ new double[n * n] {} };
    //Делим количество строк на кол-во процессов для разбиения массивов по долям
    int part = (int) (n / size); 
    //Каждый процесс работает со своими данными относительно номера исполняемого процесса
    for (unsigned int i = proc_rank * part; i < (proc_rank + 1 ) * part; i++) {
        for (int j = 0; j < n; ++j) {
            tempm[i * n + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                tempm[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double* prod{ new double[n * n] {} };
    // Все части массива по процессам объединяются в переменной prod
    MPI_Allgather(&(tempm[proc_rank * part * n]), part * n, MPI_DOUBLE, prod, part * n, MPI_DOUBLE, MPI_COMM_WORLD);
    return prod;
}

double* matcube(double*& A, int n) {
    double* B = matmul(A, A, n);
    return matmul(A, B, n);
}

double* matsquare_and_mul(double*& A, double*& B, int n){
    double* A2 = matmul(A, A, n);
    return matmul(A2, B, n);
}

// Функция для сложения двух матриц
double* matsum(double*& A, double*& B, double*& C, int n) {
    double* tempm{ new double[n * n] {} };
    for (int i = 0; i < n * n; ++i) {
        tempm[i] = A[i] + B[i] + C[i];
    }
    return tempm;
}

void printm(double * m, int n, string filename){
    std::ofstream out;                  // поток для записи
    out.open(filename);                 // открываем файл для записи
    if (out.is_open())
    {
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                out << m[i * n + j] << " ";
            }
            out << "\n";
        }
    }
    else{
        cout << "Не удалось открыть файл \n";
    }
    out.close();                      // закрываем файл
}

double* set_values(int n, int mmax, int mmin){
    double* tempm{ new double[n * n] {} };
    for (int i = 0; i < n; i++) {
        for (int j = i - 2; j < i + 3; j++) {
            if (j >= 0 && j < n) {
                tempm[i * n + j] = (double)(rand())/RAND_MAX*(mmax - mmin) + mmin;;
            }
        }
    }
    return tempm;
}

double * ones(int n){
    double* tempm{ new double[n * n] {} };
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            tempm[i * n + j] = 1;
        }
    }
    return tempm;
}

double calculate(int n, int mmax, int mmin){
    double* B = set_values(N, mmax, mmin); 
    printm(B, N, "B.txt");
    double* C = set_values(N, mmax, mmin); 
    printm(C, N, "C.txt");
    double* D = set_values(N, mmax, mmin); 
    printm(D, N, "D.txt");
    double* E = ones(N);
    printm(E, N, "E.txt");

    double tim = omp_get_wtime();
    double* B3 = matcube(B, n);
    double* C2E = matsquare_and_mul(C, E, n);
    double* BD = matmul(B, D, n);
    double* result = matsum(B3, C2E, BD, n);
    tim = omp_get_wtime()-tim;
    printm(result, n, "result.txt");
    delete[] B;
    delete[] C;
    delete[] D;
    delete[] E;
    delete[] B3;
    delete[] C2E;
    delete[] BD;
    delete[] result;
    return tim;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    double t1, t2, t3;
    cout << "Process " << proc_rank << " is now active" << endl;
    for (int i = 0; i < 4; i++) {
        N *= 2;
        if (proc_rank == 0)
            cout << "Test " << i + 1 << ", current N = " << N << endl;
        t1 = calculate(N, mmax, mmin);
        t2 = calculate(N, mmax, mmin);
        t3 = calculate(N, mmax, mmin);
        if (proc_rank == 0)
        {
            cout << t1 << " secs\n";
            cout << t2 << " secs\n";
            cout << t3 << " secs\n";
            cout << "average " << (t1 + t2 + t3) / 3 << " secs with " << size << " processes\n\n";
        }
    }

    //Удаляем структуры данных MPI
    MPI_Finalize();


    return 0;
}