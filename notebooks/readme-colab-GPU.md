# Execução no Google Colab

Ative GPU no Colab:

Ambiente de execução → Alterar tipo → GPU

Compile:

```bash
!nvcc notebooks/cuda.cu -o mlp_cuda
```

Execute:

```bash
!./mlp_cuda
```

Verificar GPU:

```bash
!nvidia-smi
```