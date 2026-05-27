import matplotlib.pyplot as plt
import pandas as pd

# 1. Organizando os dados
data = {
    "BATCH_SIZE": [1, 64, 256, 1024],
    "Código CPU (ms)": [0.1528, 10.0222, 41.7919, 163.5949],
    "Código GPU CUDA (ms)": [30.2633, 28.4229, 117.9852, 32.0763],
    "Código GPU PYTORCH (ms)": [0.1266, 0.1549, 0.1427, 0.1780],
}

# Criando o DataFrame do Pandas
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))

# 2. Plotando as linhas com os nomes das colunas EXATAMENTE iguais ao dicionário
plt.plot(
    df["BATCH_SIZE"],
    df["Código CPU (ms)"],
    marker="o",
    linewidth=2,
    label="Código CPU (sequencial)",
)

plt.plot(
    df["BATCH_SIZE"],
    df["Código GPU CUDA (ms)"],
    marker="s",
    linewidth=2,
    label="Código GPU CUDA (Customizado)",
)

plt.plot(
    df["BATCH_SIZE"],
    df["Código GPU PYTORCH (ms)"],
    marker="^",
    linewidth=2,
    label="Código GPU Pytorch (cuBLAS)",
)

# 3. Customizações de títulos e eixos
plt.title(
    "Comparativo de Tempo de Inferência MLP: CPU vs GPU Custom vs GPU PyTorch",
    fontsize=14,
    fontweight="bold",
)

plt.xlabel("Tamanho do lote (BATCH_SIZE)", fontsize=12)
plt.ylabel("Tempo de execução (ms) - Escala Logarítmica", fontsize=12)

# Ajustando as escalas logarítmicas corretamente
plt.xscale("log")
plt.xticks(df["BATCH_SIZE"], labels=[str(b) for b in df["BATCH_SIZE"]])

plt.yscale("log")

# Adicionando grade e a legenda
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=11)

plt.tight_layout()

# 4. Salvando a imagem
plt.savefig("grafico_performance_mlp.png", dpi=300)
print(
    "Gráfico gerado e salvo com sucesso como 'grafico_performance_mlp.png'!"
)

plt.show()