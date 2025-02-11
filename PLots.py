import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cargar dataset
train_df = pd.read_csv("Data/train_subset_10.csv")

# Contar clases
class_counts = train_df['label'].value_counts()

# Crear figura más grande
plt.figure(figsize=(12, 6))

# Graficar
ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')

# Agregar títulos
plt.title("Distribución de Clases", fontsize=14)
plt.xlabel("Clases", fontsize=12)
plt.ylabel("Cantidad", fontsize=12)

# Rotar los labels a 45° o 90° para evitar superposición
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=10)

# Ajustar márgenes para que no se corte
plt.tight_layout()

# Guardar imagen sin recortar
plt.savefig("class_distribution_fixed.png", bbox_inches="tight")
plt.show()
