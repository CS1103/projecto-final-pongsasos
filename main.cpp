#include <iostream>
#include <raylib.h>
#include "nn/network.h"
#include "nn/tensor.h"
#include <memory>
#include <fstream>
#include <random>

using namespace std;
using namespace utec::neural_network;
using namespace utec::algebra;

Color blue = Color{60, 110, 155};
Color dark_blue = Color{25, 30, 65};
Color light_blue = Color{110, 150, 185};

const int screen_width = 1200;
const int screen_height = 800;
int player_score = 0;
int ai_score = 0;

// Configuración de la red neuronal
const int TRAINING_GAMES = 100;
const int TRAINING_EPOCHS = 50;
bool is_training = false;
int games_played = 0;

class Ball {
public:
    float x, y;
    int speed_x, speed_y;
    int radius;

    void Draw() {
        DrawCircle(x, y, radius, WHITE);
    }

    void Update() {
        x += speed_x;
        y += speed_y;

        if (y + radius >= GetScreenHeight() || y - radius <= 0) {
            speed_y *= -1;
        }

        if (x + radius >= GetScreenWidth()) {
            ai_score++;
            Reset();
        }
        if (x - radius <= 0) {
            player_score++;
            Reset();
        }
    }

    void Reset() {
        x = GetScreenWidth()/2;
        y = GetScreenHeight()/2;
        int speed_choices[2] = {-1, 1};
        speed_x *= speed_choices[GetRandomValue(0, 1)];
        speed_y *= speed_choices[GetRandomValue(0, 1)];
    }

    // Función para obtener el estado normalizado para la IA
    vector<float> GetNormalizedState(float paddle_y) const {
        vector<float> state;
        state.push_back(x / float(screen_width));           // Posición X normalizada
        state.push_back(y / float(screen_height));          // Posición Y normalizada
        state.push_back(speed_x / 10.0f);                   // Velocidad X normalizada
        state.push_back(speed_y / 10.0f);                   // Velocidad Y normalizada
        state.push_back(paddle_y / float(screen_height));   // Posición paddle normalizada
        return state;
    }
};

class Paddle {
protected:
    void LimitMovement() {
        if (y <= 0) {
            y = 0;
        }
        if (y + height >= GetScreenHeight()) {
            y = GetScreenHeight() - height;
        }
    }

public:
    float x, y;
    float width, height;
    int speed;

    void Draw() {
        DrawRectangle(x, y, width, height, WHITE);
    }

    void Update() {
        if (IsKeyDown(KEY_UP)) {
            y -= speed;
        } else if (IsKeyDown(KEY_DOWN)) {
            y += speed;
        }
        LimitMovement();
    }
};

class AIPaddle : public Paddle {
private:
    unique_ptr<NeuralNetwork<float>> network;
    vector<vector<float>> training_data_X;
    vector<vector<float>> training_data_y;

public:
    float last_ball_x = 0;
    float action_threshold = 0.1f;  // Umbral para evitar micro-movimientos
    AIPaddle() {
        // Crear la red neuronal
        network = make_unique<NeuralNetwork<float>>();

        // Arquitectura: 5 inputs -> 16 hidden -> 16 hidden -> 1 output (más neuronas)
        network->add_dense_layer(5, 16);
        network->add_activation("tanh");
        network->add_dense_layer(16, 16);
        network->add_activation("tanh");
        network->add_dense_layer(16, 1);
        network->add_activation("tanh");

        // Configurar optimizador con learning rate más alto
        network->set_optimizer("sgd", 0.05f);
        network->set_loss_function("mse");

        cout << "Red neuronal creada con éxito!" << endl;
        network->print_architecture();
    }

    void Update(const Ball& ball) {
        if (is_training) {
            // Durante el entrenamiento, usar estrategia simple para generar datos
            UpdateTraining(ball);
        } else {
            // Durante el juego, usar la red neuronal entrenada
            UpdateWithNN(ball);
        }

        LimitMovement();
    }

    void UpdateTraining(const Ball& ball) {
        // Estrategia mejorada para generar datos de entrenamiento
        float ball_center_y = ball.y;
        float paddle_center_y = y + height/2;

        // Calcular donde debería estar el paddle
        float target_y = ball_center_y - height/2;

        // Calcular la diferencia
        float diff_y = target_y - y;

        // Normalizar la acción entre -1 y 1
        float target_action = 0.0f;
        if (abs(diff_y) > 5.0f) {  // Solo moverse si la diferencia es significativa
            target_action = std::max(-1.0f, std::min(1.0f, diff_y / (float)speed));
        }

        // Almacenar datos de entrenamiento
        vector<float> input_data = ball.GetNormalizedState(y);
        training_data_X.push_back(input_data);
        training_data_y.push_back({target_action});

        // Aplicar la acción de entrenamiento
        if (abs(diff_y) > 10.0f) {  // Umbral para evitar micro-movimientos
            if (diff_y > 0) {
                y += speed;
            } else {
                y -= speed;
            }
        }
    }

    void UpdateWithNN(const Ball& ball) {
        // Crear tensor de entrada
        vector<float> input_data = ball.GetNormalizedState(y);
        Tensor<float, 2> input_tensor(1, 5);

        for (int i = 0; i < 5; i++) {
            input_tensor(0, i) = input_data[i];
        }

        // Predecir acción
        auto prediction = network->predict(input_tensor);
        float action = prediction(0, 0);

        // Aplicar umbral para evitar micro-movimientos
        if (abs(action) > action_threshold) {
            // Escalar la acción de manera más agresiva
            float movement = action * speed * 1.5f;  // Multiplicador para movimiento más rápido

            // Aplicar movimiento discreto para evitar titubeos
            if (movement > 2.0f) {
                y += speed;
            } else if (movement < -2.0f) {
                y -= speed;
            }
        }
    }

    void TrainNetwork() {
        if (training_data_X.empty()) {
            cout << "No hay datos de entrenamiento!" << endl;
            return;
        }

        cout << "Entrenando red neuronal con " << training_data_X.size() << " ejemplos..." << endl;

        // Convertir datos a tensores
        Tensor<float, 2> X(training_data_X.size(), 5);
        Tensor<float, 2> y(training_data_y.size(), 1);

        for (size_t i = 0; i < training_data_X.size(); i++) {
            for (int j = 0; j < 5; j++) {
                X(i, j) = training_data_X[i][j];
            }
            y(i, 0) = training_data_y[i][0];
        }

        // Entrenar la red con más épocas
        network->train(X, y, TRAINING_EPOCHS * 2, true);

        cout << "Entrenamiento completado!" << endl;

        // Limpiar datos de entrenamiento
        training_data_X.clear();
        training_data_y.clear();
    }

    void SaveModel(const string& filename) {
        // Implementación básica para guardar el modelo
        // En un proyecto real, guardarías los pesos de la red
        cout << "Modelo guardado en " << filename << endl;
    }

    void LoadModel(const string& filename) {
        // Implementación básica para cargar el modelo
        // En un proyecto real, cargarías los pesos de la red
        cout << "Modelo cargado desde " << filename << endl;
    }

    // Función para ajustar el umbral de acción
    void SetActionThreshold(float threshold) {
        action_threshold = threshold;
    }
};

Ball ball;
Paddle player;
AIPaddle ai_paddle;

void ResetGame() {
    ball.Reset();
    player.y = screen_height/2 - player.height/2;
    ai_paddle.y = screen_height/2 - ai_paddle.height/2;
    player_score = 0;
    ai_score = 0;
}

void DrawUI() {
    // Dibujar scores
    DrawText(TextFormat("%i", ai_score), screen_width/4 - 20, 20, 80, WHITE);
    DrawText(TextFormat("%i", player_score), 3*screen_width/4 - 20, 20, 80, WHITE);

    // Dibujar información de entrenamiento
    if (is_training) {
        DrawText("ENTRENANDO IA...", 20, screen_height - 130, 20, YELLOW);
        DrawText(TextFormat("Juego: %i/%i", games_played, TRAINING_GAMES), 20, screen_height - 100, 20, WHITE);
        DrawText("Presiona SPACE para saltar entrenamiento", 20, screen_height - 70, 20, WHITE);
        DrawText("Presiona 'F' para entrenamiento rápido", 20, screen_height - 40, 20, WHITE);
    } else {
        DrawText("IA ENTRENADA - Presiona 'R' para re-entrenar", 20, screen_height - 70, 20, GREEN);
        DrawText("Presiona '+/-' para ajustar sensibilidad", 20, screen_height - 40, 20, WHITE);
    }

    // Instrucciones
    DrawText("Controles: UP/DOWN arrows", 20, 20, 20, WHITE);
    DrawText("'T' - Entrenar IA", 20, 50, 20, WHITE);
    DrawText("'ESC' - Salir", 20, 80, 20, WHITE);
}

int main() {
    InitWindow(screen_width, screen_height, "PONG AI");
    SetTargetFPS(60);

    // Inicializar pelota
    ball.radius = 20;
    ball.x = screen_width/2;
    ball.y = screen_height/2;
    ball.speed_x = 7;
    ball.speed_y = 7;

    // Inicializar jugador
    player.width = 25;
    player.height = 120;
    player.x = screen_width - player.width - 10;
    player.y = screen_height/2 - player.height/2;
    player.speed = 6;

    // Inicializar IA
    ai_paddle.width = 25;
    ai_paddle.height = 120;
    ai_paddle.x = 10;
    ai_paddle.y = screen_height/2 - ai_paddle.height/2;
    ai_paddle.speed = 6;

    cout << "=== PONG AI CON REDES NEURONALES ===" << endl;
    cout << "Presiona 'T' para entrenar la IA" << endl;
    cout << "Usa las flechas UP/DOWN para jugar" << endl;
    cout << "Presiona '+/-' para ajustar sensibilidad del bot" << endl;

    bool fast_training = false;

    while (!WindowShouldClose()) {
        // Manejar input
        if (IsKeyPressed(KEY_T)) {
            is_training = true;
            games_played = 0;
            fast_training = false;
            ResetGame();
            cout << "Iniciando entrenamiento..." << endl;
        }

        if (IsKeyPressed(KEY_F) && is_training) {
            fast_training = !fast_training;
            cout << "Entrenamiento rápido: " << (fast_training ? "ON" : "OFF") << endl;
        }

        if (IsKeyPressed(KEY_R) && !is_training) {
            is_training = true;
            games_played = 0;
            fast_training = false;
            ResetGame();
            cout << "Re-entrenando IA..." << endl;
        }

        if (IsKeyPressed(KEY_SPACE) && is_training) {
            is_training = false;
            ai_paddle.TrainNetwork();
            ResetGame();
        }

        // Ajustar sensibilidad del bot
        if (IsKeyPressed(KEY_KP_ADD) || IsKeyPressed(KEY_EQUAL)) {
            ai_paddle.SetActionThreshold(max(0.05f, ai_paddle.action_threshold - 0.05f));
            cout << "Sensibilidad aumentada" << endl;
        }
        if (IsKeyPressed(KEY_KP_SUBTRACT) || IsKeyPressed(KEY_MINUS)) {
            ai_paddle.SetActionThreshold(min(0.5f, ai_paddle.action_threshold + 0.05f));
            cout << "Sensibilidad disminuida" << endl;
        }

        // Lógica del juego
        if (is_training) {
            // Modo entrenamiento automático
            ball.Update();
            ai_paddle.Update(ball);

            // Entrenamiento rápido o normal
            if (fast_training) {
                SetTargetFPS(300);  // Muy rápido para entrenamiento
            } else {
                SetTargetFPS(120);  // Rápido pero visible
            }

            // Verificar si el juego terminó
            if (ai_score >= 5 || player_score >= 5) {
                games_played++;
                if (games_played >= TRAINING_GAMES) {
                    is_training = false;
                    ai_paddle.TrainNetwork();
                    cout << "¡Entrenamiento completado automáticamente!" << endl;
                }
                ResetGame();
            }
        } else {
            // Modo juego normal
            SetTargetFPS(60);
            ball.Update();
            player.Update();
            ai_paddle.Update(ball);
        }

        // Detectar colisiones
        if (CheckCollisionCircleRec(Vector2{ball.x, ball.y}, ball.radius,
                                   Rectangle{player.x, player.y, player.width, player.height})) {
            ball.speed_x *= -1;
        }

        if (CheckCollisionCircleRec(Vector2{ball.x, ball.y}, ball.radius,
                                   Rectangle{ai_paddle.x, ai_paddle.y, ai_paddle.width, ai_paddle.height})) {
            ball.speed_x *= -1;
        }

        // Dibujar
        BeginDrawing();
        ClearBackground(blue);

        // Dibujar línea central
        DrawLine(screen_width/2, 0, screen_width/2, screen_height, WHITE);

        // Dibujar elementos del juego
        ball.Draw();
        ai_paddle.Draw();
        player.Draw();

        // Dibujar UI
        DrawUI();

        EndDrawing();
    }

    CloseWindow();
    return 0;
}