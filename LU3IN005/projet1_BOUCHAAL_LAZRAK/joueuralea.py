from joueur import Joueur
from plateau import Plateau
import numpy as np



class JoueurAlea(Joueur):

    def __init__(self,couleur,nom='Unknown'):
        super().__init__(couleur,nom)

    def play(self,plateau):
        x=np.random.choice(plateau.colonnesdisponibles())#choisis une colonne aleatoire 
        plateau.play(x,self)#joue le coup
        self.nb_coup+=1
