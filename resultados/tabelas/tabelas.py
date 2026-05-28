import pandas as pd
import matplotlib.pyplot as plt

dados = {
    "Batch": [1, 64, 256, 1024],
    "CPU (ms)": [0.1528, 0.1565, 0.1632, 0.1597],
    "GPU (ms)": [30.2633, 0.4441, 0.4608, 0.0313],
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