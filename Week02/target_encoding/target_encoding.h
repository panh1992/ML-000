#pragma once
#include <map>
#include <iostream>
#include <omp.h>

using namespace std;

void target_mean(double *matrix, double *result, const long row, const long col, const long x_index, const long y_index) {

    map<double, double> value_dict;
    map<double, double> count_dict;

    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        double x = matrix[i * col + x_index];
        auto value_iter = value_dict.find(x);
        auto count_iter = count_dict.find(x);
        if(value_iter != value_dict.end()) {
            value_dict[x] = value_iter->second + matrix[i * col + y_index];
            count_dict[x] = count_iter->second + 1;
        } else {
            value_dict[x] = matrix[i * col + y_index];
            count_dict[x] = 1;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        double x = matrix[i * col + x_index];
        result[i] = (value_dict[x] - matrix[i * col + y_index]) / (count_dict[x] - 1);
    }

}
