import random 
import copy
from joueur import Joueur
from joueuralea import JoueurAlea
import numpy as np

"""

    état : Une configuration donnée du tab
    action le coup qu un joueur peut jouer (le choix d une colonne pour poser un jeton)

    L algorithme de Monte-Carlo est un algorithme probabiliste qui vise à donner une approximation 
    d un résultat trop complexe à calculer

        principe de l'algo de Monte-Carlo:
            .prend une partie en cours(les coups deja joué) l'enregistre
                .et tente plusieur partie a partir du plateau
                .d échantillonner aléatoirement et de manière uniforme l espace des possibilités  
                .rendre comme résultat la moyenne des expériences favorable au joueur de la methode
        Dans le cas d un jeu tel que le puissance 4, à un état donné jouer pour chaque action possible un
        certain nombre de parties au hasard et d en moyenner le résultat.

"""

class Monte_Carlo(JoueurAlea):


    def __init__(self,couleur,nom='Unknown',nombre_de_simulation=70):
        super().__init__(couleur,nom)
        self.nb_simu=nombre_de_simulation


    def play(self, plateau):
        #initialisation de variable
        victoire = np.zeros(plateau.C)
        totale = np.zeros(plateau.C)
        colonnesdispo = plateau.colonnesdisponibles()
        joueur1=JoueurAlea(self.couleur)
        joueur2=JoueurAlea(-self.couleur)


        for i in range(self.nb_simu):
            temp_plat = copy.deepcopy(plateau)
            
            x = random.choice(colonnesdispo)
            temp_plat.play(x, self)

            if self.has_won(plateau):
                self.nb_coup+=1
                return plateau.play(x,self)

            resultat = temp_plat.run(joueur1,joueur2)
            totale[x] += 1
            if resultat == self.couleur:
                victoire[x] += 1

        if sum(victoire) == 0:
            self.nb_coup+=1
            return plateau.play(random.choice(colonnesdispo),self)

        probs = np.divide(victoire, totale, out=np.zeros_like(victoire), where=(totale != 0))
        probs = np.array(probs)
        max_index = np.argmax(probs)
        self.nb_coup+=1
        return plateau.play(max_index, self)
