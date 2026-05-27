import pandas as pd
import matplotlib.pyplot as plt

dados = {
    "Batch": [1, 64, 256, 1024],
    "CPU (ms)": [0.1528, 10.0222, 41.7919, 163.5949],
    "GPU (ms)": [30.2633, 28.4229, 117.9852, 32.0763],
    "GPU PyTorch (ms)": [0.1266, 0.1549, 0.1427, 0.1780]
}

tabela = pd.DataFrame(dados)

fig, ax = plt.subplots(figsize=(8, 2))

ax.axis("off")

tb = ax.table(
    cellText=tabela.values,
    colLabels=tabela.columns,
    loc="center"
)

tb.auto_set_font_size(False)
tb.set_fontsize(10)
tb.scale(1.2, 1.5)

plt.savefig(
    "tabela_benchmark.png",
    bbox_inches="tight"
)

plt.show()