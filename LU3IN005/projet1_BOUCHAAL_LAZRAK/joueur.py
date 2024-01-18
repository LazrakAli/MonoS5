import numpy as np


class Joueur:
    def __init__(self, couleur,nom='Unknown'):
        self.couleur = couleur
        self.nb_coup=0
        self.nom=nom

    def play(self, plateau):
        #demande au joueuer de choisir une colonne retry toujours si invalide
        while True:
            try:
                colonne = int(input(f"{self.nom}, choisissez une colonne (0-6) pour jouer : "))
                if 0 <= colonne <= 6 and plateau.play(colonne,self):
                    self.nb_coup+=1
                    return
                else:
                    print("Colonne invalide ou déjà pleine. Choisissez une autre colonne.")
            except ValueError:
                print("Veuillez entrer un nombre entre 0 et 6.")

    def reset(self):
        self.nb_coup=0

        """determine si un joueur a gagné
        return true or false if the player won or not
    """

    def has_won(self,plateau):
        for combinaison in plateau.quadruplets_gagnant():
            if all(plateau.tab[a][b] == self.couleur for a,b in combinaison) :
                return True
        return False