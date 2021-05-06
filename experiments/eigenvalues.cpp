#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

constexpr size_t NUM_RETRIES = 7;
constexpr size_t NUM_RUNS = 100;

bool is_converged(const vector<vector<double>> &A, double rtol = 1e-6, double atol = 1e-8) {
    double total_sum = 0;
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A.size(); ++j) {
            total_sum += A[i][j] * A[i][j];

        }
    }
    double diag_sum = 0;
    for (size_t i = 0; i < A.size(); ++i) {
        diag_sum += A[i][i] * A[i][i];
    }

    return total_sum - diag_sum < rtol * total_sum + atol;
}

vector<double> dot(const vector<vector<double>> &Q, const vector<double> b) {
    vector<double> r(b.size(), 0);
    for (size_t i = 0; i < Q.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            r[i] += Q[i][j] * b[j];
        }
    }
    return r;
}

vector<vector<double>> transpose(const vector<vector<double>> &A) {
    vector<vector<double>> R = vector<vector<double>> (A[0].size(), vector<double>(A.size()));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            R[j][i] = A[i][j];
        }
    }
    return R;
}

vector<vector<double>> mult(const vector<vector<double>> &A, const vector<vector<double>> &B) {
    vector<vector<double>> R = vector<vector<double>> (A.size(), vector<double>(B[0].size(), 0));
    for (size_t i = 0; i < R.size(); ++i) {
        for (size_t j = 0; j < R[0].size(); ++j) {
            for (size_t k = 0; k < A[0].size(); ++k) {
                R[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return R;
}

vector<double> eigenvalues(vector<vector<double>> A) {
    while (!is_converged(A)) {
        vector<vector<double>> Q = transpose(A);
        vector<vector<double>> saved_Q = vector<vector<double>> (A.size(), vector<double>(A.size(), 0));

        for (size_t i = 0; i < A.size(); ++i) {
            vector<double> r = dot(saved_Q, Q[i]);

            double norm = 0;
            for (size_t j = 0; j < A.size(); ++j) {
                Q[i][j] -= r[j];
                norm += Q[i][j] * Q[i][j];
            }

            norm = sqrt(norm);

            for (size_t j = 0; j < A.size(); ++j) {
                Q[i][j] /= norm;
            }


            for (size_t j = 0; j < A.size(); ++j) {
                for (size_t k = 0 ; k < A.size(); ++k) {
                    saved_Q[j][k] += Q[i][j] * Q[i][k];
                }
            }
        }

        A = mult(Q, A);
        A = mult(A, transpose(Q));
    }

    vector<double> eigens(A.size());
    for (size_t i = 0; i < eigens.size(); ++i) {
        eigens[i] = A[i][i];
    }
    return eigens;
}

int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    if (argc != 2) {
        cout << "Usage: ./eigenvalues <filename>" << endl;
        return 0;
    }
    ifstream input(argv[1]);
    size_t n;
    input >> n;
    vector<vector<double>> A(n, vector<double>(n));
    for (auto &v : A) {
        for (auto &x : v) {
            input >> x;
        }
    }

    vector<double> x(n, 0);
    vector<double> retries;

    for (size_t i = 0; i < NUM_RETRIES; ++i) {
        unsigned long long start_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        for (size_t num = 0; num < NUM_RUNS; ++num) {
            vector<double> gg = eigenvalues(A);
            for (size_t xx = 0; xx < n; ++xx) {
                x[xx] += gg[xx];
            }
        }
        unsigned long long end_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        retries.push_back((end_time - start_time) / 1e6 / NUM_RUNS);
    }

    for (auto x : retries) {
        cout << x << " ";
    }
    cout << endl;
}
