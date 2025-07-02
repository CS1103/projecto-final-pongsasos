#include <iostream>
#include <raylib.h>
using namespace std;

Color blue = Color{60, 110, 155};
Color dark_blue = Color{25, 30, 65};
Color light_blue = Color{110, 150, 185};

const int screen_width = 1200;
const int screen_height = 800;
int player_score = 0;
int nn_pong_score = 0;

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
            nn_pong_score++;
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
        int speed_choices[2] = {-1,1};
        speed_x *= speed_choices[GetRandomValue(0,1)];
        speed_y *= speed_choices[GetRandomValue(0,1)];
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

class PongPaddle : public Paddle {
public:
    void Update(int ball_y) {
        // aquí integrar redes neuronales :O
        if (y + height/2 > ball_y) {
            y -= speed;
        }
        if (y + height/2 <= ball_y) {
            y += speed;
        }
        LimitMovement();
    }
};

Ball ball;
Paddle player;
PongPaddle nn_pong;

int main() {
    InitWindow(screen_width, screen_height, "!!! PONGSASOOO");
    SetTargetFPS(60);

    ball.radius = 20;
    ball.x = screen_width/2;
    ball.y = screen_height/2;
    ball.speed_x = 7;
    ball.speed_y = 7;

    player.width = 25;
    player.height = 120;
    player.x = screen_width - player.width - 10;
    player.y = screen_height/2 - player.height/2;
    player.speed = 6;

    nn_pong.width = 25;
    nn_pong.height = 120;
    nn_pong.x = 10;
    nn_pong.y = screen_height/2 - nn_pong.height/2;
    nn_pong.speed = 6;

    while (!WindowShouldClose()) {
        BeginDrawing();

        ball.Update();
        player.Update();
        nn_pong.Update(ball.y);

        if (CheckCollisionCircleRec(Vector2(ball.x, ball.y),
                                    ball.radius,
                                    Rectangle(player.x, player.y, player.width, player.height))) {
            ball.speed_x *= -1;
        }

        if (CheckCollisionCircleRec(Vector2(ball.x, ball.y),
                                    ball.radius,
                                    Rectangle(nn_pong.x, nn_pong.y, nn_pong.width, nn_pong.height))) {
            ball.speed_x *= -1;
        }

        ClearBackground(blue);
        DrawLine(screen_width/2, 0, screen_width/2, screen_height, WHITE);
        ball.Draw();
        nn_pong.Draw();
        player.Draw();
        DrawText(TextFormat("%i", nn_pong_score),screen_width/4 - 20, 20, 80, WHITE);
        DrawText(TextFormat("%i", player_score), 3*screen_width/4 - 20, 20, 80, WHITE);
        EndDrawing();
    }

    CloseWindow();

    return 0;
}