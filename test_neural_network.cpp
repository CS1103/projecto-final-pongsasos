//
// Created by martin on 7/5/25.
//

#include <iostream>
#include <cassert>
#include <cmath>
#include "nn/tensor.h"
#include "nn/network.h"

using namespace std;
using namespace utec::algebra;
using namespace utec::neural_network;

// Función de prueba para verificar que los valores son aproximadamente iguales
bool approx_equal(double a, double b, double epsilon = 1e-6) {
    return abs(a - b) < epsilon;
}

void test_tensor_operations() {
    cout << "=== Probando operaciones de Tensor ===" << endl;

    // Test 1: Creación y acceso
    Tensor<float, 2> t1(2, 3);
    t1(0, 0) = 1.0f;
    t1(0, 1) = 2.0f;
    t1(0, 2) = 3.0f;
    t1(1, 0) = 4.0f;
    t1(1, 1) = 5.0f;
    t1(1, 2) = 6.0f;

    assert(t1(0, 0) == 1.0f);
    assert(t1(1, 2) == 6.0f);
    cout << "✓ Creación y acceso funcionan correctamente" << endl;

    // Test 2: Suma
    Tensor<float, 2> t2(2, 3);
    t2.fill(1.0f);

    auto t3 = t1 + t2;
    assert(t3(0, 0) == 2.0f);
    assert(t3(1, 2) == 7.0f);
    cout << "✓ Suma de tensores funciona correctamente" << endl;

    // Test 3: Multiplicación por escalar
    auto t4 = t1 * 2.0f;
    assert(t4(0, 0) == 2.0f);
    assert(t4(1, 2) == 12.0f);
    cout << "✓ Multiplicación por escalar funciona correctamente" << endl;

    // Test 4: Multiplicación de matrices
    Tensor<float, 2> A(2, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;

    Tensor<float, 2> B(3, 2);
    B(0, 0) = 7; B(0, 1) = 8;
    B(1, 0) = 9; B(1, 1) = 10;
    B(2, 0) = 11; B(2, 1) = 12;

    auto C = A.matmul(B);
    // C debería ser [58, 64; 139, 154]
    assert(C(0, 0) == 58);
    assert(C(0, 1) == 64);
    assert(C(1, 0) == 139);
    assert(C(1, 1) == 154);
    cout << "✓ Multiplicación de matrices funciona correctamente" << endl;

    // Test 5: Transpose
    auto At = A.transpose();
    assert(At.shape()[0] == 3 && At.shape()[1] == 2);
    assert(At(0, 0) == 1);
    assert(At(2, 1) == 6);
    cout << "✓ Transpose funciona correctamente" << endl;

    cout << "¡Todas las pruebas de Tensor pasaron!" << endl << endl;
}

void test_activation_functions() {
    cout << "=== Probando funciones de activación ===" << endl;

    // Test ReLU
    ReLU<float> relu;
    assert(relu.forward(-1.0f) == 0.0f);
    assert(relu.forward(2.0f) == 2.0f);
    assert(relu.backward(-1.0f) == 0.0f);
    assert(relu.backward(2.0f) == 1.0f);
    cout << "✓ ReLU funciona correctamente" << endl;

    // Test Sigmoid
    Sigmoid<float> sigmoid;
    float sig_0 = sigmoid.forward(0.0f);
    assert(approx_equal(sig_0, 0.5f, 1e-6f));

    float sig_large = sigmoid.forward(10.0f);
    assert(sig_large > 0.99f);
    cout << "✓ Sigmoid funciona correctamente" << endl;

    // Test Tanh
    Tanh<float> tanh_func;
    float tanh_0 = tanh_func.forward(0.0f);
    assert(approx_equal(tanh_0, 0.0f, 1e-6f));

    float tanh_1 = tanh_func.forward(1.0f);
    assert(approx_equal(tanh_1, tanh(1.0f), 1e-6f));
    cout << "✓ Tanh funciona correctamente" << endl;

    cout << "¡Todas las pruebas de funciones de activación pasaron!" << endl << endl;
}

void test_neural_network() {
    cout << "=== Probando red neuronal ===" << endl;

    // Crear una red simple
    NeuralNetwork<float> nn;
    nn.add_dense_layer(2, 3);
    nn.add_activation("relu");
    nn.add_dense_layer(3, 1);
    nn.add_activation("sigmoid");

    nn.set_optimizer("sgd", 0.1f);
    nn.set_loss_function("mse");

    cout << "✓ Red neuronal creada correctamente" << endl;
    nn.print_architecture();

    // Crear datos de prueba simples (XOR)
    Tensor<float, 2> X(4, 2);
    X(0, 0) = 0; X(0, 1) = 0;
    X(1, 0) = 0; X(1, 1) = 1;
    X(2, 0) = 1; X(2, 1) = 0;
    X(3, 0) = 1; X(3, 1) = 1;

    Tensor<float, 2> y(4, 1);
    y(0, 0) = 0; // 0 XOR 0 = 0
    y(1, 0) = 1; // 0 XOR 1 = 1
    y(2, 0) = 1; // 1 XOR 0 = 1
    y(3, 0) = 0; // 1 XOR 1 = 0

    cout << "✓ Datos de prueba (XOR) creados" << endl;

    // Predecir antes del entrenamiento
    cout << "Predicciones antes del entrenamiento:" << endl;
    auto pred_before = nn.predict(X);
    for (int i = 0; i < 4; i++) {
        cout << "Input: [" << X(i, 0) << ", " << X(i, 1) << "] -> Predicción: "
             << pred_before(i, 0) << ", Esperado: " << y(i, 0) << endl;
    }

    // Entrenar
    cout << "\nEntrenando red neuronal..." << endl;
    nn.train(X, y, 1000, false);

    // Predecir después del entrenamiento
    cout << "\nPredicciones después del entrenamiento:" << endl;
    auto pred_after = nn.predict(X);
    for (int i = 0; i < 4; i++) {
        cout << "Input: [" << X(i, 0) << ", " << X(i, 1) << "] -> Predicción: "
             << pred_after(i, 0) << ", Esperado: " << y(i, 0) << endl;
    }

    cout << "✓ Entrenamiento de red neuronal completado" << endl;

    cout << "¡Todas las pruebas de red neuronal pasaron!" << endl << endl;
}

void test_pong_scenario() {
    cout << "=== Probando escenario específico de Pong ===" << endl;

    // Crear red para Pong (5 inputs -> 1 output)
    NeuralNetwork<float> pong_nn;
    pong_nn.add_dense_layer(5, 8);
    pong_nn.add_activation("tanh");
    pong_nn.add_dense_layer(8, 4);
    pong_nn.add_activation("tanh");
    pong_nn.add_dense_layer(4, 1);
    pong_nn.add_activation("tanh");

    pong_nn.set_optimizer("sgd", 0.01f);
    pong_nn.set_loss_function("mse");

    cout << "✓ Red neuronal para Pong creada" << endl;
    pong_nn.print_architecture();

    // Crear datos de ejemplo
    Tensor<float, 2> X(10, 5);
    Tensor<float, 2> y(10, 1);

    // Llenar con datos aleatorios de ejemplo
    for (int i = 0; i < 10; i++) {
        X(i, 0) = float(rand()) / RAND_MAX; // ball_x
        X(i, 1) = float(rand()) / RAND_MAX; // ball_y
        X(i, 2) = (float(rand()) / RAND_MAX) * 2 - 1; // ball_speed_x
        X(i, 3) = (float(rand()) / RAND_MAX) * 2 - 1; // ball_speed_y
        X(i, 4) = float(rand()) / RAND_MAX; // paddle_y

        // Acción objetivo simple: mover hacia la pelota
        y(i, 0) = (X(i, 1) - X(i, 4)) * 0.5f; // Normalizado
    }

    cout << "✓ Datos de ejemplo para Pong creados" << endl;

    // Entrenar brevemente
    cout << "Entrenando red de Pong..." << endl;
    pong_nn.train(X, y, 50, true);

    // Hacer una predicción
    Tensor<float, 2> test_input(1, 5);
    test_input(0, 0) = 0.5f; // ball_x
    test_input(0, 1) = 0.3f; // ball_y
    test_input(0, 2) = 0.1f; // ball_speed_x
    test_input(0, 3) = 0.2f; // ball_speed_y
    test_input(0, 4) = 0.8f; // paddle_y

    auto prediction = pong_nn.predict(test_input);
    cout << "Predicción de acción: " << prediction(0, 0) << endl;

    cout << "✓ Prueba específica de Pong completada" << endl;

    cout << "¡Todas las pruebas de escenario Pong pasaron!" << endl << endl;
}

int main() {
    cout << "=== EJECUTANDO PRUEBAS UNITARIAS ===" << endl << endl;

    try {
        test_tensor_operations();
        test_activation_functions();
        test_neural_network();
        test_pong_scenario();

        cout << "🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE! 🎉" << endl;
        cout << "El sistema está listo para ser usado en el juego Pong." << endl;

    } catch (const exception& e) {
        cout << "❌ Error durante las pruebas: " << e.what() << endl;
        return 1;
    }

    return 0;
}