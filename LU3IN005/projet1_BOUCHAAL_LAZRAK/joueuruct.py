from monte_carlo import Monte_Carlo
from joueuralea import JoueurAlea
import numpy as np
import random
import copy

class JoueurUCT(JoueurAlea):
    """
    Description de PlayerUCT:

    .Contrairement à une exploration complètement aléatoire de toutes les possibilités, UCT priorise 
    l'exploration des actions les plus prometteuses pour atteindre la victoire tout en continuant à 
    explorer d'autres options possibles

    .Pour résoudre ce dilemme, l'algorithme UCB est utilisé à chaque embranchement 
    possible pour équilibrer l'exploration et l'exploitation.

    .la racine représentant l'état actuel du jeu.

    .pour explorer rapidement la qualité d'un état, stratégie par défaut est la stratégie aléatoire

    .lorsqu'aucune information n'est disponible, des simulations avec un joueur aléatoire sont 
    effectuées pour chaque action afin d'initialiser les noeuds enfants de la racine.

    .les nœuds enfants correspondant à chaque action sont créés pour le moment, 
    et les états visités pendant la simulation ne sont pas stockés

    .Chaque noeud développé stocke le nombre de fois où il a conduit à une victoire et le nombre de fois où 
    il a été joué dans une simulation (victoire, défaite ou match nul)


    """
    def __init__(self,couleur,nom='Unknown',nombre_de_simulation=100,debut=30):
        super().__init__(couleur,nom)
        self.nb_simu=nombre_de_simulation
        self.debut=debut

    def meilleure_action_UCB(self, estimations, nb_visites, i):
        # Fonction pour déterminer la meilleure action en utilisant l'algorithme UCB
        if np.any(nb_visites == 0):
            my_func = np.zeros(len(estimations))
        else:
            exploration = np.sqrt(2 * np.log(i) / nb_visites)
            my_func = estimations + exploration
        return np.argmax(my_func)



    def play(self,plateau):
        #initialisation des variables:
        nb_visites = np.zeros(plateau.C)#nombre de visites de chaque action,chaque element correspond à une action spécifique et représente le nombre de fois que cette action a été sélectionnée jusqu'à présent
        estimations = np.zeros(plateau.C)#liste qui maintient une estimation de la valeur, chaque element correspond à une action spécifique et représente une estimation de la probabilité de gagner ou de la valeur de cette action
        resultat=np.zeros(self.nb_simu)#stocke les résultats des simulations effectuées à chaque itération, chaque element correspond à un résultat de simulation pour une action spécifique
        actions=[] #représente le coup choisi à l'itération correspondante
        
        #variables classique (MTC):
        victoire = np.zeros(plateau.C)
        totale = np.zeros(plateau.C)
        colonnesdispo = plateau.colonnesdisponibles()
        joueur1 = JoueurAlea(couleur=-self.couleur)
        joueur2 = JoueurAlea(couleur=self.couleur)

        for i in range(self.nb_simu):
            temp_plat=copy.deepcopy(plateau)
            if i<self.debut:
                actions.append(random.choice(colonnesdispo))
            else:
                actions.append(self.meilleure_action_UCB(estimations,nb_visites,i-1))

            nb_visites[actions[i]] += 1 #indentation de la visite de l'action 
            temp_plat.play(actions[i], self)#placer la piece

            #classique: si on gagne on arrete
            if self.has_won(temp_plat):
                self.nb_coup+=1
                return plateau.play(actions[i],self)
            resultat = temp_plat.run(joueur1,joueur2)
            totale[actions[i]] += 1
            if resultat == self.couleur:
                victoire[actions[i]] += 1
            if totale[actions[i]] != 0: #met a jour la proba
                estimations[actions[i]]=victoire[actions[i]]/totale[actions[i]]
        if sum(victoire)==0:
            self.nb_coup+=1
            return plateau.play(random.choice(colonnesdispo),self)
        self.nb_coup+=1
        return plateau.play(np.argmax(estimations),self)#joue le coup avec l'indice le plus grand