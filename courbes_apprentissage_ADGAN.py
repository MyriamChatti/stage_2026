import numpy as np
import matplotlib.pyplot as plt

file_path = "/home/myriam/Documents/stage_M2/Stage_2026/code/ADGAN/logs/output_2026-02-06_13-46-07.csv"
data = np.genfromtxt(file_path, delimiter=",")
data = data[~np.isnan(data).any(axis=1)]

# Colonnes
lr     = data[:, 0]
g_rec  = data[:, 1]
g_cyc  = data[:, 2]
g_ctr  = data[:, 3]
adv    = data[:, 4]
dis    = data[:, 6]

iterations = np.arange(1, len(g_rec) + 1)

# figure
plt.figure(figsize=(10,6))

plt.plot(iterations, g_rec, label="Reconstruction")
plt.plot(iterations, g_cyc, label="Cycle")
plt.plot(iterations, g_ctr, label="Contrastive")
plt.plot(iterations, adv, label="Adversarial")
plt.plot(iterations, dis,   label="Discriminator")

plt.xlabel("Iterations")
plt.ylabel("Loss value")
plt.title("ADGAN - Learning Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("courbes d'apprentissage ADGAN.png", dpi=300)
plt.close()

print("succès.")