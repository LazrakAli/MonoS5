import sys
from joueur import Joueur
from plateau import Plateau
from joueuralea import JoueurAlea

class Partie:

    """
        initialise une partie
            plateau
            2 joueurs
            et le joueurs qui doit jouer
    """
    def __init__(self,j1=JoueurAlea(1),j2=JoueurAlea(-1)):
        self.plateau = Plateau()
        self.joueur1 = j1
        self.joueur2 = j2


    """ 
        run sous format: (joueur,jeu)
        resultat format:   
    """
    def run_format_de_lecture(self):
        resultat=self.plateau.run(self.joueur1,self.joueur2)
        if resultat==self.joueur1.couleur:
            print(f"{self.joueur1.nb_coup},{resultat}")
        elif resultat==self.joueur2.couleur:
            print(f"{self.joueur2.nb_coup},{resultat}")
        else:
            cpt=self.plateau.C*self.plateau.L
            print(f"{cpt},{resultat}")



    """
        ecrit le resultat de 1000 tests dans le fichier test.txt
    """
    def ecriture_dans_un_fichier(self, nom):
        original_stdout = sys.stdout#sauvegarde la sortie de base
        with open(f'{nom}.txt', 'w') as file:#ouvre le fichier
            sys.stdout = file#swap les sortie
            i=0
            while i<100:
                self.plateau.reset()
                self.joueur1.reset()
                self.joueur2.reset()
                self.run_format_de_lecture()
                i+=1
        sys.stdout = original_stdout #remet la sortie de base

