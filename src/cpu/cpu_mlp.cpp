#include <stdio.h>
#include <stdlib.h>
#include <chrono> // Para medição de tempo de alta precisão na CPU

#define INPUT_SIZE 784
#define HIDDEN_1   128
#define HIDDEN_2   64
#define OUTPUT_SIZE 10
#define BATCH_SIZE 1024 // Mude para 1, 64, 256, 1024 igual na GPU

// Função de ativação ReLU na CPU
float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

// Inicialização aleatória idêntica ao código da GPU
void init_random(float* arr, int size, float scale) {
    for (int i = 0; i < size; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * scale;
    }
}

int main() {
    printf("=== Inicializando Processamento CPU para MLP ===\n");
    printf("Configuração atual: BATCH_SIZE = %d\n\n", BATCH_SIZE);

    // Alocação de memória na CPU
    float *X   = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    float *W1  = (float*)malloc(HIDDEN_1 * INPUT_SIZE * sizeof(float));
    float *b1  = (float*)malloc(HIDDEN_1 * sizeof(float));
    float *Z1  = (float*)malloc(BATCH_SIZE * HIDDEN_1 * sizeof(float));

    float *W2  = (float*)malloc(HIDDEN_2 * HIDDEN_1 * sizeof(float));
    float *b2  = (float*)malloc(HIDDEN_2 * sizeof(float));
    float *Z2  = (float*)malloc(BATCH_SIZE * HIDDEN_2 * sizeof(float));

    float *W3  = (float*)malloc(OUTPUT_SIZE * HIDDEN_2 * sizeof(float));
    float *b3  = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float *Out = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    srand(42);
    init_random(X, BATCH_SIZE * INPUT_SIZE, 1.0f);
    init_random(W1, HIDDEN_1 * INPUT_SIZE, 0.01f); init_random(b1, HIDDEN_1, 0.0f);
    init_random(W2, HIDDEN_2 * HIDDEN_1, 0.01f);  init_random(b2, HIDDEN_2, 0.0f);
    init_random(W3, OUTPUT_SIZE * HIDDEN_2, 0.1f);  init_random(b3, OUTPUT_SIZE, 0.0f);

    // --- INÍCIO DA MEDIÇÃO DE TEMPO NA CPU ---
    auto start = std::chrono::high_resolution_clock::now();

    // Camada 1: Entrada -> Oculta 1 (Linear + ReLU)
    for (int row = 0; row < BATCH_SIZE; ++row) {
        for (int col = 0; col < HIDDEN_1; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < INPUT_SIZE; ++k) {
                sum += X[row * INPUT_SIZE + k] * W1[col * INPUT_SIZE + k];
            }
            Z1[row * HIDDEN_1 + col] = relu(sum + b1[col]);
        }
    }

    // Camada 2: Oculta 1 -> Oculta 2 (Linear + ReLU)
    for (int row = 0; row < BATCH_SIZE; ++row) {
        for (int col = 0; col < HIDDEN_2; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < HIDDEN_1; ++k) {
                sum += Z1[row * HIDDEN_1 + k] * W2[col * HIDDEN_1 + k];
            }
            Z2[row * HIDDEN_2 + col] = relu(sum + b2[col]);
        }
    }

    // Camada 3: Oculta 2 -> Saída (Apenas Linear)
    for (int row = 0; row < BATCH_SIZE; ++row) {
        for (int col = 0; col < OUTPUT_SIZE; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < HIDDEN_2; ++k) {
                sum += Z2[row * HIDDEN_2 + k] * W3[col * HIDDEN_2 + k];
            }
            Out[row * OUTPUT_SIZE + col] = sum + b3[col];
        }
    }

    // --- FIM DA MEDIÇÃO DE TEMPO NA CPU ---
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;

    printf("Tempo puro de processamento na CPU: %.4f ms\n", duration.count());

    // Liberar memória alocada
    free(X);
    free(W1);
    free(b1);
    free(Z1);
    free(W2);
    free(b2);
    free(Z2);
    free(W3);
    free(b3);
    free(Out);

    return 0;
}
