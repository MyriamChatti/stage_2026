import csv
import os






def save_as_csv(csv_path, values):
    #Sauvegarde une liste de valeurs numériques dans un CSV (une ligne par itération)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(values)





#--------------------------------------------------------------------
# quelques fonctions csv

def check_directories(dir_A, dir_B):

    #je vérifie que les dossiers A et B existent.

    assert os.path.isdir(dir_A), f"Dossier introuvable : {dir_A}"
    assert os.path.isdir(dir_B), f"Dossier introuvable : {dir_B}"


def get_file_lists(dir_A, dir_B):

    #je récupère et trie les fichiers des dossiers A et B.

    files_A = sorted(os.listdir(dir_A))
    files_B = sorted(os.listdir(dir_B))
    return files_A, files_B


def save_pairs_to_csv(dir_A, dir_B, out_csv):

    #je sauvegarde les paires (A, B) dans un fichier CSV.

    files_A, files_B = get_file_lists(dir_A, dir_B)

    assert len(files_A) == len(files_B), \
        "A et B n'ont pas le même nombre de fichiers"




def test_csv(out_csv, n_lines=5):
    #j'affiche les premières lignes du CSV pour vérification.
    print("\n--- Test du CSV ---")
    with open(out_csv, mode="r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            print(row)
            if i >= n_lines:
                break




#ouvre fichier csv
def save_as_csv(out_csv, list_to_save):
    with open(out_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list_to_save)

        #for a, b in zip(files_A, files_B):
        #    writer.writerow([
         #       os.path.join(dir_A, a),
          #      os.path.join(dir_B, b)
         #   ])
#ls logs/2026-01-27T10-04-15_ADGAN/checkpoints
#(adgan) myriam@FSPP25020:~/Documents/stage_M2/Stage_2026/code/ADGAN$ 






# --------------------------------------------------
# la phase test avec main
if __name__ == "__main__":
    
    #dir_A = "datasets/YourDATA/train/A"
    #dir_B = "datasets/YourDATA/train/B"
    dirc = "logs/output/"
    out_csv = "logs/output.csv"

    # phase vérification
    check_directories(dir_A, dir_B)

    # création du CSV
    save_pairs_to_csv(dir_A, dir_B, out_csv)

    # phase test
    test_csv(out_csv)






