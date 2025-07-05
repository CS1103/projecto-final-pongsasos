#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "tensor.h"
#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

namespace utec {
namespace neural_network {

// Funciones de activación
template<typename T>
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual T forward(T x) const = 0;
    virtual T backward(T x) const = 0;
    virtual std::string name() const = 0;
};

template<typename T>
class ReLU : public ActivationFunction<T> {
public:
    T forward(T x) const override {
        return std::max(T{0}, x);
    }
    
    T backward(T x) const override {
        return x > T{0} ? T{1} : T{0};
    }
    
    std::string name() const override { return "relu"; }
};

template<typename T>
class Tanh : public ActivationFunction<T> {
public:
    T forward(T x) const override {
        return std::tanh(x);
    }
    
    T backward(T x) const override {
        T tanh_x = std::tanh(x);
        return T{1} - tanh_x * tanh_x;
    }
    
    std::string name() const override { return "tanh"; }
};

template<typename T>
class Sigmoid : public ActivationFunction<T> {
public:
    T forward(T x) const override {
        return T{1} / (T{1} + std::exp(-x));
    }
    
    T backward(T x) const override {
        T sig_x = forward(x);
        return sig_x * (T{1} - sig_x);
    }
    
    std::string name() const override { return "sigmoid"; }
};

// Capas de la red neuronal
template<typename T>
class Layer {
public:
    virtual ~Layer() = default;
    virtual utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) = 0;
    virtual utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad_output) = 0;
    virtual void update_weights(T learning_rate) {}
    virtual std::string type() const = 0;
};

template<typename T>
class DenseLayer : public Layer<T> {
private:
    utec::algebra::Tensor<T, 2> weights_;
    utec::algebra::Tensor<T, 2> biases_;
    utec::algebra::Tensor<T, 2> last_input_;
    utec::algebra::Tensor<T, 2> weight_gradients_;
    utec::algebra::Tensor<T, 2> bias_gradients_;
    
public:
    DenseLayer(size_t input_size, size_t output_size) 
        : weights_(input_size, output_size), biases_(1, output_size) {
        
        // Inicialización Xavier
        std::random_device rd;
        std::mt19937 gen(rd());
        T limit = std::sqrt(T{6} / (input_size + output_size));
        std::uniform_real_distribution<T> dis(-limit, limit);
        
        for (size_t i = 0; i < input_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                weights_(i, j) = dis(gen);
            }
        }
        
        biases_.fill(T{0});
    }
    
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) override {
        last_input_ = input;
        
        // output = input * weights + biases
        auto output = input.matmul(weights_);
        
        // Agregar bias a cada fila
        for (size_t i = 0; i < output.shape()[0]; ++i) {
            for (size_t j = 0; j < output.shape()[1]; ++j) {
                output(i, j) += biases_(0, j);
            }
        }
        
        return output;
    }
    
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad_output) override {
        // Calcular gradientes de los pesos
        weight_gradients_ = last_input_.transpose().matmul(grad_output);
        
        // Calcular gradientes de los biases
        bias_gradients_ = utec::algebra::Tensor<T, 2>(1, grad_output.shape()[1]);
        for (size_t j = 0; j < grad_output.shape()[1]; ++j) {
            T sum = T{0};
            for (size_t i = 0; i < grad_output.shape()[0]; ++i) {
                sum += grad_output(i, j);
            }
            bias_gradients_(0, j) = sum;
        }
        
        // Calcular gradientes para la capa anterior
        return grad_output.matmul(weights_.transpose());
    }
    
    void update_weights(T learning_rate) override {
        // Actualizar pesos
        for (size_t i = 0; i < weights_.shape()[0]; ++i) {
            for (size_t j = 0; j < weights_.shape()[1]; ++j) {
                weights_(i, j) -= learning_rate * weight_gradients_(i, j);
            }
        }
        
        // Actualizar biases
        for (size_t j = 0; j < biases_.shape()[1]; ++j) {
            biases_(0, j) -= learning_rate * bias_gradients_(0, j);
        }
    }
    
    std::string type() const override { return "dense"; }
};

template<typename T>
class ActivationLayer : public Layer<T> {
private:
    std::unique_ptr<ActivationFunction<T>> activation_;
    utec::algebra::Tensor<T, 2> last_input_;
    
public:
    ActivationLayer(const std::string& activation_name) {
        if (activation_name == "relu") {
            activation_ = std::make_unique<ReLU<T>>();
        } else if (activation_name == "tanh") {
            activation_ = std::make_unique<Tanh<T>>();
        } else if (activation_name == "sigmoid") {
            activation_ = std::make_unique<Sigmoid<T>>();
        } else {
            throw std::invalid_argument("Unknown activation function: " + activation_name);
        }
    }
    
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) override {
        last_input_ = input;
        return input.apply([this](T x) { return activation_->forward(x); });
    }
    
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad_output) override {
        auto grad_input = last_input_.apply([this](T x) { return activation_->backward(x); });
        
        // Element-wise multiplication
        utec::algebra::Tensor<T, 2> result(grad_output.shape()[0], grad_output.shape()[1]);
        for (size_t i = 0; i < grad_output.shape()[0]; ++i) {
            for (size_t j = 0; j < grad_output.shape()[1]; ++j) {
                result(i, j) = grad_output(i, j) * grad_input(i, j);
            }
        }
        
        return result;
    }
    
    std::string type() const override { return "activation_" + activation_->name(); }
};

// Red neuronal completa
template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer<T>>> layers_;
    T learning_rate_;
    std::string optimizer_;
    std::string loss_function_;
    
    T calculate_loss(const utec::algebra::Tensor<T, 2>& predictions, 
                    const utec::algebra::Tensor<T, 2>& targets) {
        if (loss_function_ == "mse") {
            T loss = T{0};
            size_t total_elements = predictions.shape()[0] * predictions.shape()[1];
            
            for (size_t i = 0; i < predictions.shape()[0]; ++i) {
                for (size_t j = 0; j < predictions.shape()[1]; ++j) {
                    T diff = predictions(i, j) - targets(i, j);
                    loss += diff * diff;
                }
            }
            
            return loss / total_elements;
        }
        
        throw std::invalid_argument("Unknown loss function: " + loss_function_);
    }
    
    utec::algebra::Tensor<T, 2> calculate_loss_gradient(const utec::algebra::Tensor<T, 2>& predictions,
                                                       const utec::algebra::Tensor<T, 2>& targets) {
        if (loss_function_ == "mse") {
            utec::algebra::Tensor<T, 2> grad(predictions.shape()[0], predictions.shape()[1]);
            T scale = T{2} / (predictions.shape()[0] * predictions.shape()[1]);
            
            for (size_t i = 0; i < predictions.shape()[0]; ++i) {
                for (size_t j = 0; j < predictions.shape()[1]; ++j) {
                    grad(i, j) = scale * (predictions(i, j) - targets(i, j));
                }
            }
            
            return grad;
        }
        
        throw std::invalid_argument("Unknown loss function: " + loss_function_);
    }
    
public:
    NeuralNetwork() : learning_rate_(T{0.001}), optimizer_("sgd"), loss_function_("mse") {}
    
    void add_dense_layer(size_t input_size, size_t output_size) {
        layers_.push_back(std::make_unique<DenseLayer<T>>(input_size, output_size));
    }
    
    void add_activation(const std::string& activation_name) {
        layers_.push_back(std::make_unique<ActivationLayer<T>>(activation_name));
    }
    
    void set_optimizer(const std::string& optimizer, T learning_rate) {
        optimizer_ = optimizer;
        learning_rate_ = learning_rate;
    }
    
    void set_loss_function(const std::string& loss_function) {
        loss_function_ = loss_function;
    }
    
    utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& input) {
        utec::algebra::Tensor<T, 2> output = input;
        
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        
        return output;
    }
    
    void train(const utec::algebra::Tensor<T, 2>& X, const utec::algebra::Tensor<T, 2>& y, 
               int epochs, bool verbose = true) {
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Forward pass
            auto predictions = predict(X);
            
            // Calculate loss
            T loss = calculate_loss(predictions, y);
            
            if (verbose && epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
            }
            
            // Backward pass
            auto grad_output = calculate_loss_gradient(predictions, y);
            
            // Backpropagate through all layers
            for (int i = layers_.size() - 1; i >= 0; --i) {
                grad_output = layers_[i]->backward(grad_output);
            }
            
            // Update weights
            for (auto& layer : layers_) {
                layer->update_weights(learning_rate_);
            }
        }
    }
    
    void print_architecture() {
        std::cout << "Neural Network Architecture:" << std::endl;
        for (size_t i = 0; i < layers_.size(); ++i) {
            std::cout << "Layer " << i << ": " << layers_[i]->type() << std::endl;
        }
    }
};

} // namespace neural_network
} // namespace utec

#endif // NN_NETWORK_H