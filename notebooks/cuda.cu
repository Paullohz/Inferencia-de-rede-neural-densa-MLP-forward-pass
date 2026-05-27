#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Definição da arquitetura da MLP conforme o enunciado
#define INPUT_SIZE 784   // 28x28 pixels
#define HIDDEN_1   128   // Neurônios da Camada Oculta 1
#define HIDDEN_2   64    // Neurônios da Camada Oculta 2
#define OUTPUT_SIZE 10   // Classes de saída (0 a 9)

// Tamanho do lote (batch)
#define BATCH_SIZE 1024

// Função auxiliar para verificar erros do CUDA
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        printf("Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError())); \
        exit(1); \
    }

/**
 * Kernel CUDA Fusor: Multiplicação de Matrizes (Linear) + Bias + ReLU
 * Cada thread calcula um elemento da matriz de saída (linha do batch, coluna do neurônio).
 * Operação matemática: Out[row, col] = max(0, sum(X[row, k] * W[col, k]) + b[col])
 */
__global__ void mlp_layer_linear_relu_kernel(
    const float* __restrict__ X,  // Matriz de Entrada (BATCH_SIZE x in_features)
    const float* __restrict__ W,  // Matriz de Pesos (out_features x in_features)
    const float* __restrict__ b,  // Vetor de Bias (out_features)
    float* __restrict__ Out,      // Matriz de Saída (BATCH_SIZE x out_features)
    int in_features,
    int out_features)
{
    // Mapeamento Bidimensional de Threads
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Índice da imagem no Batch
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Índice do neurônio de saída

    if (row < BATCH_SIZE && col < out_features) {
        float sum = 0.0f;

        // Produto escalar entre a linha da entrada e a linha do peso (W está em Row-Major out x in)
        for (int k = 0; k < in_features; ++k) {
            sum += X[row * in_features + k] * W[col * in_features + k];
        }

        // Fusão do Bias
        sum += b[col];

        // Fusão da Função de Ativação ReLU: f(x) = max(0, x)
        Out[row * out_features + col] = (sum > 0.0f) ? sum : 0.0f;
    }
}

/**
 * Kernel para a Camada de Saída (Apenas Linear, sem ReLU)
 */
__global__ void mlp_layer_linear_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ Out,
    int in_features,
    int out_features)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < BATCH_SIZE && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += X[row * in_features + k] * W[col * in_features + k];
        }
        Out[row * out_features + col] = sum + b[col]; // Sem ReLU na última camada
    }
}

// Função simples para preencher matrizes com valores aleatórios (Alternativa 1)
void init_random(float* arr, int size, float scale) {
    for (int i = 0; i < size; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * scale;
    }
}

int main() {
    printf("=== Inicializando Acelerador GPU (CUDA) para MLP ===\n");
    printf("Configuração atual: BATCH_SIZE = %d\n\n", BATCH_SIZE);

    // 1. Alocação de memória no Host (CPU)
    size_t bytes_X  = BATCH_SIZE * INPUT_SIZE * sizeof(float);
    size_t bytes_W1 = HIDDEN_1 * INPUT_SIZE * sizeof(float);
    size_t bytes_b1 = HIDDEN_1 * sizeof(float);
    size_t bytes_Z1 = BATCH_SIZE * HIDDEN_1 * sizeof(float);

    size_t bytes_W2 = HIDDEN_2 * HIDDEN_1 * sizeof(float);
    size_t bytes_b2 = HIDDEN_2 * sizeof(float);
    size_t bytes_Z2 = BATCH_SIZE * HIDDEN_2 * sizeof(float);

    size_t bytes_W3 = OUTPUT_SIZE * HIDDEN_2 * sizeof(float);
    size_t bytes_b3 = OUTPUT_SIZE * sizeof(float);
    size_t bytes_Out= BATCH_SIZE * OUTPUT_SIZE * sizeof(float);

    float *h_X  = (float*)malloc(bytes_X);
    float *h_W1 = (float*)malloc(bytes_W1); float *h_b1 = (float*)malloc(bytes_b1);
    float *h_W2 = (float*)malloc(bytes_W2); float *h_b2 = (float*)malloc(bytes_b2);
    float *h_W3 = (float*)malloc(bytes_W3); float *h_b3 = (float*)malloc(bytes_b3);
    float *h_Out= (float*)malloc(bytes_Out); // Guardará o resultado final vindo da GPU

    // Inicialização dos pesos e entradas
    srand(42);
    init_random(h_X, BATCH_SIZE * INPUT_SIZE, 1.0f);
    init_random(h_W1, HIDDEN_1 * INPUT_SIZE, 0.01f); init_random(h_b1, HIDDEN_1, 0.0f);
    init_random(h_W2, HIDDEN_2 * HIDDEN_1, 0.01f);  init_random(h_b2, HIDDEN_2, 0.0f);
    init_random(h_W3, OUTPUT_SIZE * HIDDEN_2, 0.1f);  init_random(h_b3, OUTPUT_SIZE, 0.0f);

    // 2. Alocação de memória no Device (GPU)
    float *d_X, *d_W1, *d_b1, *d_Z1, *d_W2, *d_b2, *d_Z2, *d_W3, *d_b3, *d_Out;
    CUDA_CHECK(cudaMalloc(&d_X, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_W1, bytes_W1)); CUDA_CHECK(cudaMalloc(&d_b1, bytes_b1));
    CUDA_CHECK(cudaMalloc(&d_Z1, bytes_Z1));
    CUDA_CHECK(cudaMalloc(&d_W2, bytes_W2)); CUDA_CHECK(cudaMalloc(&d_b2, bytes_b2));
    CUDA_CHECK(cudaMalloc(&d_Z2, bytes_Z2));
    CUDA_CHECK(cudaMalloc(&d_W3, bytes_W3)); CUDA_CHECK(cudaMalloc(&d_b3, bytes_b3));
    CUDA_CHECK(cudaMalloc(&d_Out, bytes_Out));

    // Eventos CUDA para medição precisa de tempo na GPU
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_total));   CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_kernel));  CUDA_CHECK(cudaEventCreate(&stop_kernel));

    // --- INÍCIO DA MEDIÇÃO TOTAL (Inclui Cópia H2D, Execução e Cópia D2H) ---
    CUDA_CHECK(cudaEventRecord(start_total));

    // 3. Cópia de dados do Host para o Device (H2D)
    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1, bytes_W1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1, bytes_b1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2, bytes_W2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2, bytes_b2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, h_W3, bytes_W3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3, bytes_b3, cudaMemcpyHostToDevice));

    // --- INÍCIO DA MEDIÇÃO SÓ DOS KERNELS ---
    CUDA_CHECK(cudaEventRecord(start_kernel));

    // 4. Configuração da Grade de Blocos (Grid and Block Dimensions)
    dim3 threadsPerBlock(16, 16);

    // Camada 1
    dim3 numBlocks1((HIDDEN_1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (BATCH_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mlp_layer_linear_relu_kernel<<<numBlocks1, threadsPerBlock>>>(d_X, d_W1, d_b1, d_Z1, INPUT_SIZE, HIDDEN_1);

    // Camada 2
    dim3 numBlocks2((HIDDEN_2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (BATCH_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mlp_layer_linear_relu_kernel<<<numBlocks2, threadsPerBlock>>>(d_Z1, d_W2, d_b2, d_Z2, HIDDEN_1, HIDDEN_2);

    // Camada 3 (Saída - Sem ReLU)
    dim3 numBlocks3((OUTPUT_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (BATCH_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mlp_layer_linear_kernel<<<numBlocks3, threadsPerBlock>>>(d_Z2, d_W3, d_b3, d_Out, HIDDEN_2, OUTPUT_SIZE);

    // --- FIM DA MEDIÇÃO SÓ DOS KERNELS ---
    CUDA_CHECK(cudaEventRecord(stop_kernel));

    // 5. Cópia dos resultados de volta do Device para o Host (D2H)
    CUDA_CHECK(cudaMemcpy(h_Out, d_Out, bytes_Out, cudaMemcpyDeviceToHost));

    // --- FIM DA MEDIÇÃO TOTAL ---
    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    // Calcular e exibir tempos decorridos
    float ms_total = 0, ms_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start_total, stop_total));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, start_kernel, stop_kernel));

    printf("Tempo total de Execução (com Transferências): %.4f ms\n", ms_total);
    printf("Tempo puro de processamento dos Kernels:     %.4f ms\n", ms_kernel);
    printf("Inferência na GPU finalizada com sucesso.\n\n");

    // Print de validação (primeiros 5 resultados da primeira imagem do batch)
    printf("Primeiras 5 saídas do neurônio para a primeira imagem do lote:\n");
    for(int i = 0; i < 5; i++) {
        printf("Classe [%d]: %.4f\n", i, h_Out[i]);
    }

    // 6. Limpeza de Memória (Free) - LINHAS CORRIGIDAS AQUI
    free(h_X); free(h_W1); free(h_b1);
    free(h_W2); free(h_b2); free(h_W3); free(h_b3); free(h_Out);

    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_W1)); CUDA_CHECK(cudaFree(d_b1)); CUDA_CHECK(cudaFree(d_Z1));
    CUDA_CHECK(cudaFree(d_W2)); CUDA_CHECK(cudaFree(d_b2)); CUDA_CHECK(cudaFree(d_Z2));
    CUDA_CHECK(cudaFree(d_W3)); CUDA_CHECK(cudaFree(d_b3)); CUDA_CHECK(cudaFree(d_Out));

    CUDA_CHECK(cudaEventDestroy(start_total)); CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_kernel)); CUDA_CHECK(cudaEventDestroy(stop_kernel));

    return 0;
}
