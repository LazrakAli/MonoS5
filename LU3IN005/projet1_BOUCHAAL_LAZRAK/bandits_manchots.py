import random 
import math

"""
On dispose d'une machine à sous avec N leviers, chacun ayant un rendement défini par un paramètre µi (probabilité de succès) constant.
À chaque instant t, le joueur choisit un levier à actionner, noté at.
Lorsque le joueur actionne un levier at, il obtient une récompense Rt qui suit une distribution de Bernoulli avec le paramètre µ^at.
Le but est de maximiser le gain total après T actions, où le gain total GT est la somme des récompenses obtenues lors des T actions.

"""

class Bandits_Manchots:
    #l'ensemble des leviers est représenté par la liste des parametre de la loi de Bernouli
    def __init__(self,rendement):
        self.rendement=rendement
        self.N=len(rendement)
        self.estimations=[0]*self.N
        self.n_joue=[0]*self.N
        self.total_joue=0

    
    def jouer_coup(self, action):
        if action < 0 or action >= self.N:
            raise ValueError("L'indice du Levier n'existe pas")
        
        rendement_choisi = self.rendement[action]
        #génération d'une récomp binaire selon une distribution de Bernoulli
        recompense = 1 if random.random() < rendement_choisi else 0
        self.total_joue+=1
        return recompense

    #choisis un coup aleatoirement de façon uniforme
    def choisir_coup_alea(self):
        return random.randint(0,self.N-1)

    """
    ajouter un random sur le quelle faire le tampon
    """
    def choisir_coup_greedy(self,nb_ite,nb_exploration=None):
        if nb_ite<self.N:
            raise ValueError(f"Le nombre d'itérations doit être supérieure {self.N}")
        if nb_exploration>nb_ite:
            raise ValueError("Le nombre d'exploration doit être inferieur au nombre d'iteration")
        if nb_exploration==None:
            nb_exploration=self.N #on prend par default une exploration uniforme des N premiers leviers
        for i in range(nb_ite):
            if i<nb_exploration : # on fait tout d'abord de l'exploration pour les N premiers coups
                action=self.choisir_coup_alea()
            else:
                action = self.estimations.index(max(self.estimations)) #exploitation choisir le levier avec le rendement estimé maximal
            
            #Joue le coup et obtenir la récompense
            recompense = self.jouer_coup(action)
            
            #Met a jour les stats
            self.n_joue[action] += 1
            self.estimations[action] += (recompense - self.estimations[action]) / self.n_joue[action]
        return self.estimations.index(max(self.estimations)) #retourne le liver avec le redement max


    """
        idée principale derrière l'algorithme ϵ-greedy est de combiner à la fois l'exploration et l'exploitation.
        fonctionnement:
            1)l'algorithme effectuera une exploration aléatoire, en choisissant n'importe quelle action 
            de manière équitable parmi les actions possibles. Cela permet d'explorer de nouvelles actions.
            2)Phase d'exploitation : Avec une probabilité 1 - ϵ, l'algorithme choisira l'action qui a le rendement estimé le plus élevé 
            jusqu'à présent. 
            Cela signifie qu'il exploitera l'action qui semble être la meilleure en fonction des informations collectées jusqu'à présent.

            avantage theorique:
                .réglant ϵ à une valeur plus élevée, encourager davantage l'exploration
                .une valeur plus faible, privilégier davantage l'exploitation. 
                .permet de trouver un bon équilibre entre l'apprentissage et la prise de décision basée sur les informations disponibles
                .devient de plus en plus orienté vers l'exploitation des actions ayant de bons rendements estimés ( collecte plus d'informations (à travers des itérations) )
    """ 
    def choisir_coup_ep_greedy(self,nb_ite,epsilon):
        if nb_ite<=0 or epsilon<0 or epsilon>=1:
            raise ValueError("Choisir un nombre d'iteration superieur à 0 et un epsilon entre [0-1[")
        for i in range(nb_ite):
            if random.random()<epsilon:
                action=self.choisir_coup_alea()
            else :
                action = self.estimations.index(max(self.estimations))
            recompense=self.jouer_coup(action)
            self.n_joue[action] += 1
            self.estimations[action] += (recompense - self.estimations[action]) / self.n_joue[action]
        action=self.estimations.index(max(self.estimations))
        return action

    """
        Le premier terme est identique aux autres algorithmes, il garantit l exploitation.
        le deuxième  le ratio entre le nombre de coups total et le nombre de fois où une action donnée a été choisie devient grand

    """
    
    def choisir_coup_UCB(self, nb_iteration, nb_exploration=None, epsilon=None):
        if epsilon is not None and nb_exploration is not None:
            raise ValueError("Le choix de l'exploration doit être soit epsilon soit nb_exploration, pas les deux.")
        
        if epsilon is not None:
            self.choisir_coup_ep_greedy(nb_iteration, epsilon)
        else:
            self.choisir_coup_greedy(nb_iteration, nb_exploration)
        
        ucb_values = []
        for i in range(self.N):
            exploration_bonus = math.sqrt(2 * math.log(sum(self.n_joue)) / self.n_joue[i])
            ucb = self.estimations[i] + exploration_bonus
            ucb_values.append(ucb)
        
        action = ucb_values.index(max(ucb_values))
        
        # Mettre à jour les compteurs
        recompense = self.jouer_coup(action)
        self.n_joue[action] += 1
        self.estimations[action] += (recompense - self.estimations[action])
        
        return action
        

            


"""
#test
BM=Bandits_Manchots([0.8, 0.5, 0.6, 0.3])
print(BM.jouer_coup(BM.choisir_coup_UCB(100430,500)))
print(f"le nombre totale de coupe joué est {BM.total_joue}")
"""