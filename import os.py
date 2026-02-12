import os
import csv


####fonctions traitant les csv ou aussi créer classes 

def saving():

assert len(files_A) == len(files_B), "A et B n'ont pas le même nombre d'images"

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["A", "B"])  # pour noms des colonnes
    for a, b in zip(files_A, files_B):
        writer.writerow([
            os.path.join(dir_A, a),
            os.path.join(dir_B, b)
        ])

print("CSV créé :", out_csv)
print("Nombre de lignes :", len(files_A))


##########main
if __name__ == '__main__':
    dir_A = "datasets/YourDATA/train/A"
    dir_B = "datasets/YourDATA/train/B"
    out_csv = "datasets/YourDATA/train/output.csv"

files_A = sorted(os.listdir(dir_A))
files_B = sorted(os.listdir(dir_B))

