import numpy as np
from joueur import Joueur
from colorama import Fore, Back, Style, init

class Plateau:

    """
        constructeur: initialisant le tableau avec array de la bilitheque numpy et le remplissant de 0
    """
    def __init__(self,lignes=6,colonnes=7):
        self.L=lignes
        self.C=colonnes
        self.tab = np.zeros((self.L, self.C))



    # Fonction pour afficher le plateau de jeu avec les jetons colorés ou "o"
    def afficher_plateau(self):
        for ligne in self.tab:
            ligne_str = "|"
            for case in ligne:
                if case == 1:
                    # Jeton joueur 1 (rouge)
                    ligne_str += Back.RED + "○" + Style.RESET_ALL + "|"
                elif case == -1:
                    # Jeton joueur 2 (bleu)
                    ligne_str += Back.BLUE + "○" + Style.RESET_ALL + "|"
                else:
                    # Cas vide
                    ligne_str += "o|"
            print(ligne_str)
        print("+-----------------------------+")


    """ fonction enumerant toute les combinaison gagnante"""
    def quadruplets_gagnant(self):
        combinaisons=[]

        #combiaisons par ligne
        for i in range(self.L):
            for j in range(self.C - 3):
                combinaisons.append(((i, j), (i, j+1), (i, j+2), (i, j+3)))

        #combinaisons par colonne
        for i in range(self.L - 3):
            for j in range(self.C):
                combinaisons.append(((i, j), (i+1, j), (i+2, j), (i+3, j)))

        #combinaisons diagonales descendante
        for i in range(self.L - 3):
            for j in range(self.C - 3):
                combinaisons.append(((i, j), (i+1, j+1), (i+2, j+2), (i+3, j+3)))


        #combinaisons diagonales montantes
        for i in range(3, self.L):
            for j in range(self.C - 3):
                combinaisons.append(((i, j), (i-1, j+1), (i-2, j+2), (i-3, j+3)))

        return combinaisons

    """fonction qui vide le tab"""
    def reset(self):
        self.tab = np.zeros_like(self.tab)#cree un tab de meme taille que l'originale en vide



    """
        le joeurs place un jeton de coordonnée colonne x dans le tab
        return True si on réussi à placer le jeton, False sinon
    """
    def play(self,x,joueur):
        if self.tab[0][x]!=0:
            return False #toute la colonne et deja pleine

        for ligne in range(self.L-1,-1,-1): # on commence par la derniere ligne de coordonné L-1 et on decremente jusqu'a trouver une case vide on prend la case 0 qui est la dernier ligne
            if self.tab[ligne][x]==0:
                self.tab[ligne][x]=joueur.couleur
                return True #tout c'est bien passé
        return False


    """
    verifie si le jeu est fini 2 cas:
        toute les case sont pleine -> true
        qlq a gagné joueur1 ou joueur2 -> true
    dans le cas suivant false

    """
    def is_finished(self,*joueurs: Joueur):
 
        #regarde si toute les cases sont pleine
        if all(self.tab[0][c] != 0 for c in range(self.C)):
            return 0

        # Vérifie si l'un des joueurs a gagné
        if joueurs[0].has_won(self):
            return joueurs[0].couleur

        if joueurs[1].has_won(self):
            return joueurs[1].couleur
        # Si aucune des conditions n'est remplie, le jeu n'est pas encore fini

        return None

    """
        retourne les colonnes disponible
    """
    def colonnesdisponibles(self):
        colonnesdispo=[]
        for i in range(self.C):
            if self.tab[0][i]==0:
                colonnesdispo.append(i)
        return colonnesdispo


    def run(self,*joueurs: Joueur,affichage_complet=False):
        
        if len(joueurs) != 2:
            raise ValueError("La méthode run doit être appelée avec exactement deux joueurs.")
        
        while True:
            if affichage_complet:
                self.afficher_plateau()  # Affiche le plateau
            for joueur in joueurs:

                resultat = self.is_finished(*joueurs)

                if resultat == 0:
                    if affichage_complet:
                        self.afficher_plateau() 
                        print("La partie est un match nul.")
                    return 0
                elif resultat == 1:
                    if affichage_complet:
                        self.afficher_plateau()
                        print(f"{joueurs[0].nom} a gagné !")
                    return 1
                elif resultat == -1:
                    if affichage_complet:
                        self.afficher_plateau()
                        print(f"{joueur1[1].nom} a gagné !")
                    return -1
                else :
                    joueur.play(self)  # Le joueur actuel joue son tour
                self.afficher_plateau()


