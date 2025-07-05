//
// Created by martin on 7/5/25.
//

#ifndef NN_TENSOR_H
#define NN_TENSOR_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cmath>
#include <iostream>
#include <random>

namespace utec {
namespace algebra {

template<typename T, size_t N>
class Tensor {
private:
    std::vector<T> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    void calculate_strides() {
        strides_.resize(N);
        if (N > 0) {
            strides_[N-1] = 1;
            for (int i = N-2; i >= 0; --i) {
                strides_[i] = strides_[i+1] * shape_[i+1];
            }
        }
    }

    size_t get_index(const std::vector<size_t>& indices) const {
        size_t idx = 0;
        for (size_t i = 0; i < N; ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            idx += indices[i] * strides_[i];
        }
        return idx;
    }

public:
    // Constructor por defecto
    Tensor() : shape_(N, 0), strides_(N, 0) {}

    // Constructor con forma específica
    template<typename... Args>
    Tensor(Args... dimensions) : shape_{static_cast<size_t>(dimensions)...} {
        static_assert(sizeof...(dimensions) == N, "Number of dimensions must match template parameter");

        size_t total_size = 1;
        for (size_t dim : shape_) {
            total_size *= dim;
        }

        data_.resize(total_size);
        calculate_strides();
        std::fill(data_.begin(), data_.end(), T{});
    }

    // Constructor con inicialización
    template<typename... Args>
    Tensor(T init_value, Args... dimensions) : shape_{static_cast<size_t>(dimensions)...} {
        static_assert(sizeof...(dimensions) == N, "Number of dimensions must match template parameter");

        size_t total_size = 1;
        for (size_t dim : shape_) {
            total_size *= dim;
        }

        data_.resize(total_size, init_value);
        calculate_strides();
    }

    // Acceso a elementos (para 2D)
    T& operator()(size_t i, size_t j) {
        static_assert(N == 2, "This operator is only for 2D tensors");
        return data_[get_index({i, j})];
    }

    const T& operator()(size_t i, size_t j) const {
        static_assert(N == 2, "This operator is only for 2D tensors");
        return data_[get_index({i, j})];
    }

    // Acceso a elementos (para 1D)
    T& operator()(size_t i) {
        static_assert(N == 1, "This operator is only for 1D tensors");
        return data_[i];
    }

    const T& operator()(size_t i) const {
        static_assert(N == 1, "This operator is only for 1D tensors");
        return data_[i];
    }

    // Métodos de información
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }

    // Operaciones matemáticas
    Tensor<T, N> operator+(const Tensor<T, N>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }

        Tensor<T, N> result = *this;
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] += other.data_[i];
        }
        return result;
    }

    Tensor<T, N> operator-(const Tensor<T, N>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for subtraction");
        }

        Tensor<T, N> result = *this;
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] -= other.data_[i];
        }
        return result;
    }

    Tensor<T, N> operator*(T scalar) const {
        Tensor<T, N> result = *this;
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] *= scalar;
        }
        return result;
    }

    // Multiplicación de matrices (solo para 2D)
    Tensor<T, 2> matmul(const Tensor<T, 2>& other) const {
        static_assert(N == 2, "Matrix multiplication is only for 2D tensors");

        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Invalid dimensions for matrix multiplication");
        }

        Tensor<T, 2> result(shape_[0], other.shape_[1]);

        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < other.shape_[1]; ++j) {
                T sum = T{};
                for (size_t k = 0; k < shape_[1]; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }

        return result;
    }

    // Aplicar función a todos los elementos
    template<typename Func>
    Tensor<T, N> apply(Func func) const {
        Tensor<T, N> result = *this;
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = func(data_[i]);
        }
        return result;
    }

    // Transponer (solo para 2D)
    Tensor<T, 2> transpose() const {
        static_assert(N == 2, "Transpose is only for 2D tensors");

        Tensor<T, 2> result(shape_[1], shape_[0]);
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // Llenar con valores aleatorios
    void random_fill(T min_val = T{}, T max_val = T{1}) {
        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dis(min_val, max_val);
            for (size_t i = 0; i < data_.size(); ++i) {
                data_[i] = dis(gen);
            }
        } else {
            std::uniform_int_distribution<T> dis(min_val, max_val);
            for (size_t i = 0; i < data_.size(); ++i) {
                data_[i] = dis(gen);
            }
        }
    }

    // Llenar con un valor específico
    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Acceso directo a los datos
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // Imprimir tensor (para debugging)
    void print() const {
        if (N == 1) {
            std::cout << "[";
            for (size_t i = 0; i < shape_[0]; ++i) {
                std::cout << (*this)(i);
                if (i < shape_[0] - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } else if (N == 2) {
            std::cout << "[\n";
            for (size_t i = 0; i < shape_[0]; ++i) {
                std::cout << "  [";
                for (size_t j = 0; j < shape_[1]; ++j) {
                    std::cout << (*this)(i, j);
                    if (j < shape_[1] - 1) std::cout << ", ";
                }
                std::cout << "]";
                if (i < shape_[0] - 1) std::cout << ",";
                std::cout << "\n";
            }
            std::cout << "]" << std::endl;
        }
    }
};

} // namespace algebra
} // namespace utec

#endif //NN_TENSOR_H