################################## [LU3IN005] PROJET 1: BATAILLE NAVALE ###########################################

'''
    Le jeu de la bataille navale se joue sur une grille de 10 cases par 10 cases. L’adversaire place sur cette 
grille un certain nombre de bateaux qui sont caractérisés par leur longueur :
    • un porte-avions (5 cases)
    • un croiseur (4 cases)
    • un contre-torpilleurs (3 cases)
    • un sous-marin (3 cases)
    • un torpilleur (2 cases)

Il a le droit de les placer verticalement ou horizontalement. Le positionnement des bateaux reste secret, 
l’objectif du joueur est de couler tous les bateaux de l’adversaire en un nombre minimum de coup. A chaque tour 
de jeu, il choisit une case où il tire : si il n’y a aucun bateau sur la case, la réponse est vide ; si la case est 
occupée par une partie d’un bateau, la réponse est touché ; si toutes les cases d’un bateau ont été touchées,
alors la réponse est coulé et les cases correspondantes sont révélées. Lorsque tous les bateaux ont été coulés, 
le jeu s’arrête et le score du joueur est le nombre de coups qui ont été joués. 
Plus le score est petit, plus le joueur est performant.
'''

################################### PARTIE 1: MODÉLISATION ET FONCTIONS SIMPLES ###################################

import math
import numpy as np
import matplotlib.pyplot as plt
import copy

import random as rd

### CONSTANTES ET VARIABLES


# Dimensions de la grille
DIMX = 10
DIMY = 10

''' On crée un dictionnaire où chaque bateau est caractérisé par un identifiant chiffré 
(sa clé) et a pour valeur le nombre de cases qu'il occupe dans la grille:
    • un porte-avion (1): 5 cases
    • un croiseur (2): 4 cases
    • un contre-torpilleurs (3): 3 cases
    • un sous-marin (4): 3 cases
    • un torpilleur (5): 2 cases
'''

dic_bateaux = {1: 5, 2: 4, 3: 3, 4: 3, 5: 2}


def creer_grille():
    """ void-> matrice
    Renvoie une matrice d'entiers initialisés à 0 de dimension DIMX*DIMY
    """
    return np.zeros((DIMX,DIMY), dtype=int)

def creer_grille_float():
    """ void-> matrice
    Renvoie une matrice de flottants initialisés à 0 de dimension DIMX*DIMY
    """
    return np.zeros((DIMX,DIMY), dtype=float)



def peut_placer(grille, bateau, position, direction):
    """ Matrice,int,(int,int),char -> bool
    Renvoie true si la tête de bateau est placable dans la grille
    Sinon renvoie False"""
   
    placable = True
    
    if(direction == 'h'): #direction horizontale
        if(position[1] + dic_bateaux[bateau]-1 > DIMY-1): 
            placable = False #Cas où les coordonnées dépassent
        else:
            for i in range(dic_bateaux[bateau]):
                if(grille[position[0], position[1] + i] != 0):
                    placable = False
    
    elif(direction == 'v'): #direction verticale
        if(position[0] + dic_bateaux[bateau]-1 > DIMX-1): 
            placable = False #Cas où les coordonnées dépassent
        else:
            for i in range(dic_bateaux[bateau]):
                if(grille[position[0] + i, position[1]] != 0):
                    placable = False
    else:
        print("Erreur: direction non valide")
        return None
    
    return placable





def placer(grille, bateau, position, direction):
    """Matrice,int,(int,int),char -> void
    Modifie la grille en ajoutant le bateau si c'est placable
    """

    #Vérifie au préalable que l'on peut placer le bateau
    if (peut_placer(grille, bateau, position, direction) == False) :
        print("Erreur: placement non valide")
        return 

    if(direction == 'h'):
        for i in range(dic_bateaux[bateau]):
            grille[position[0], position[1] + i] = bateau
    
    elif(direction == 'v'):
        for i in range(dic_bateaux[bateau]):
            grille[position[0] + i, position[1]] = bateau
    else:
        print("Erreur: direction non valide")
        return None
    
    return 





def placer_alea(grille, bateau):
    """Matrice,int -> (int,int),char
    Renvoie la position et la direction de bateau aléatoirement choisi
    Modifie la grille avec le placement de bateau """
    
    estPlace = False
    position = None
    direction = None
    
    while(estPlace == False):
        #Gérer cas boucle infinie avec coompteur par ex
        position = (rd.randint(0,DIMX-1), rd.randint(0,DIMY-1))
        if(rd.random() < 0.5): direction = 'h'
        else: direction = 'v'
        
        estPlace = peut_placer(grille, bateau, position, direction)
    
    placer(grille, bateau, position, direction)
    return position, direction




from matplotlib import pyplot as plt
from matplotlib import colors

def affiche(grille):
    """ Matrice->void
    Affiche la grille """
    fig = plt.figure(figsize=(6,6))
    fig.suptitle("---------- BATAILLE NAVALE ----------\n")
    plt.pcolor(grille, edgecolors='w', linewidth=0.5)
    plt.show()

def affiche_valeurs(grille, coord_objet_perdu, coord_senseur):
    """ Matrice->void
    Affiche la grille avec ses valeurs par case"""
    fig, ax = plt.subplots()


    ax.matshow(grille, cmap=plt.cm.Blues)

    for i in range(DIMX):
        for j in range(DIMY):
            c = grille[i,j]
            if ((i,j) == coord_senseur):
                ax.text(i, j, f"{c:.2f}", va='center', ha='center', color = 'red')
            elif ((i,j) == coord_objet_perdu):
                ax.text(i, j, f"{c:.2f}", va='center', ha='center', color = 'green')
            else:
                ax.text(i, j, f"{c:.2f}", va='center', ha='center')

    ax.invert_yaxis()
    plt.xticks(np.arange(0, DIMX, 1.0))
    plt.gca().xaxis.tick_bottom()
    plt.yticks(np.arange(0, DIMY, 1.0))  

    plt.show(block=False)
    plt.pause(1)
    plt.close()
    

def eq(grilleA, grilleB):
    """Matrice,Matrice -> bool
    Renvoie vrai si les matrices sont équivalents."""
    if(len(grilleA)!=len(grilleB)):
        return False
    
    for i in range(DIMX):
        if(len(grilleA[i])!=len(grilleB[i])):
            return False
        for j in range(DIMY):
            if grilleA[i][j] != grilleB[i][j]:
                return False
    
    return True




def grille_copie(grille):
    """Matrice->Matrice
    Renvoie la copie generée de la matrice en argument"""
    copie=copy.deepcopy(grille)
    return copie
            

  

def genere_grille(liste_bateaux = dic_bateaux.keys()):
    """List[int]->matrice,List[int] 
    Genere une grille  pour un joueur avec les bateaux dans la liste en argument
    Renvoie la liste avec les positions des bateaux et la grille generé pour 1 seul joueur"""
    grille = creer_grille()
    
    # Liste des positions et directions de bateaux générés pour chaque joueur
    positions_bateaux = []
    
    # On place les bateaux des 2 joueurs
    for id_bateau in liste_bateaux:
        position, direction = placer_alea(grille, id_bateau)
        for i in range(dic_bateaux[id_bateau]):
            x, y = position
            if direction == 'h':
                positions_bateaux.append((x, y + i))
            else:
                positions_bateaux.append((x + i, y))
    return grille, positions_bateaux

def MaxPointAPrendre(dic=dic_bateaux):
    """dict->int
    Renvoie le valer possible qu'un joueur peut acquérir avec la liste des bateaux dans la grille"""
    return sum(dic.values())



# --------------------------------------------- TESTS INTERMÉDIAIRES ---------------------------------------------

def testTME1():
    grille, positions_bateaux = genere_grille()
    affiche(grille)

#testTME1()



########################################## PARTIE 2: COMBINATOIRE DU JEU ##########################################


''' QUESTION 1: Donner une borne supérieure du nombre de configurations possibles pour la liste complète 
                de bateaux sur une grille de taille 10 (calcul à la main).

           
RÉPONSE:
    
    Comme l'on ne veut qu'une borne supérieure du nombre de configurations possibles pour la liste complète 
                de bateaux sur la grille, on prendra aussi en compte les cas où des bateaux sont superposés
                (ce que l'on ne devrait normalement pas faire).
                
    Ainsi POUR CHAQUE JOUEUR:
    
        * Placer un bateau de 5 cases (5 cases côte à côte):
           Sur une ligne ou une colonne: (10 - 5) + 1 = 6 possibilités. 
           --> 6*20 = 120 (horizontal et vertical) façons de placer un porte-avion dans la grille (10 lignes, 10 colonnes)
                
        * Puis placer un bateau de 4 cases (4 cases côte à côte):
            Sur une ligne ou une colonne: (10 - 4) + 1 = 7 possibilités.
            --> 7*20 = 140 (horizontal et vertical) façons de placer un croiseur dans la grille (10 lignes, 10 colonnes)
        
        * Puis placer un bateau de 3 cases (3 cases côte à côte):
            Sur une ligne ou une colonne: (10 - 3) + 1 = 8 possibilités.
            --> 8*20 = 160 (horizontal et vertical) façons de placer un contre-torpilleur dans la grille (10 lignes, 10 colonnes)
            
        * Puis placer un bateau de 3 cases (3 cases côte à côte):
            Sur une ligne ou une colonne: (10 - 3) + 1 = 8 possibilités.
            --> 8*20 = 160 (horizontal et vertical) façons de placer un sous-marin dans la grille (10 lignes, 10 colonnes)
            
        * Puis placer un bateau de 2 cases (2 cases côte à côte):
            Sur une ligne/colonne: (10 - 2) + 1 = 9 possibilités.
            --> 9*20 = 180 (horizontal et vertical) façons de placer un torpilleur dans la grille (10 lignes, 10 colonnes)
        
    ---> Il y a donc 120 * 140 * 160 * 160 * 180 = 77 414 400 000 configurations possibles de la grille pour 1 joueur
'''



''' QUESTION 2: Donner d’abord une fonction qui permet de dénombrer le nombre de façons de placer
                un bateau donné sur une grille vide. Comparer au résultat théorique
'''


def question2(id_bateau):
    """int->int
    Renvoie le nombre des différents façon de bateau en arguement """
    grille = creer_grille()
    cpt = 0
    for i in range(DIMX):
        for j in range(DIMY):
            if peut_placer(grille, id_bateau, (i,j), 'h'):
                cpt +=1
            if peut_placer(grille, id_bateau, (i,j), 'v'):
                cpt +=1
    return cpt

### Comparaison avec la réponse donnée à la question 1


# assert(question2(1) == 120)
# assert(question2(2) == 140)
# assert(question2(3) == 160)
# assert(question2(4) == 160)
# assert(question2(5) == 180)

# print("QUESTION 2: Le test est un succès.")




# print("Il y a...")
# print("\t", question2(1), " façons de placer un porte-avion de 5 cases")
# print("\t", question2(2), " façons de placer un croiseur de 4 cases")
# print("\t", question2(3), " façons de placer un contre-torpilleur de 3 cases")
# print("\t", question2(4), " façons de placer un sous-marin de 3 cases")
# print("\t", question2(5), " façons de placer un torpilleur de 2 cases")



''' QUESTION 3: Donner une fonction qui permet de dénombrer le nombre de façon de placer une liste
                de bateaux sur une grille vide. Calculer le nombre de grilles différentes pour 1, 2 et 3
                bateaux. Est-il possible de calculer de cette manière le nombre de grilles pour la liste
                complète de bateau ?
'''

''' Retourne le nombre de façon de placer une liste de bateaux sur une grille vide
'''

def question3(liste_bateaux, grille):
    """List[int],Matrice->int
    Renvoie le nombre des différents façon de placement des bateaux dans la liste """
        
    #COMMANDE: question3(liste_bateaux, creer_grille())
    
    if(len(liste_bateaux) == 0):
        return 0
    
    cpt = 0
    bateau = liste_bateaux[0]
    
    if(len(liste_bateaux) == 1):
        
        cpt = question2(bateau)

    if(len(liste_bateaux) > 1):
        
        for i in range(DIMX):
            for j in range(DIMY):
                
                if peut_placer(grille, bateau, (i,j), 'h'):
                    copie=grille_copie(grille)
                    placer(copie,bateau,(i,j),'h')
                    cpt += question3(liste_bateaux[1:], copie)
                    
                if peut_placer(grille, bateau, (i,j), 'v'):
                    copie=grille_copie(grille)
                    placer(copie,bateau,(i,j),'v')
                    cpt += question3(liste_bateaux[1:],copie)
    
    return cpt



#print("\nQUESTION 3:")
#print("\t\t Nombre de grilles pour une liste de 1 bateau [1]: ", question3([1], creer_grille()))
#print("\t\t Nombre de grilles pour une liste de 2 bateau [1,2]:", question3([1,2], creer_grille()))
#print("\t\t Nombre de grilles pour une liste de 3 bateau [1,2,3]:", question3([1,2,3], creer_grille()))



''' Est-il possible de calculer de cette manière le nombre de grilles pour la liste
complète de bateau ?

RÉPONSE:
    
    Il est possible de calculer de cette manière le nombre de grilles pour la liste
complète de bateau, bien que le calcul prendra beaucoup de temps à s'exécuter.
'''



''' QUESTION 4: En considérant toutes les grilles équiprobables, quel est le lien entre le nombre de
                grilles et la probabilité de tirer une grille donnée ? Donner une fonction qui prend en
                paramètre une grille, génère des grilles aléatoirement jusqu’à ce que la grille générée
                soit égale à la grille passée en paramètre et qui renvoie le nombre de grilles générées.
'''

'''
RÉPONSE:
    Si toutes les grilles sont équiprobables, alors soit p la probabilité de tirer une grille donnée et n le
    nombre de grille. Alors p = 1/n.
'''

def question4(grille):
    """Matrice->int
    Le nombre de repete pour avoir la même matrice par rapport de matrice en arguent"""
    cpt = 0
    while eq(grille, genere_grille()[0]) == False:
        cpt += 1
    return cpt



#print("\nQUESTION 4:")
#print("\t Nombre de tirages avant d'obtenir une grille générée aléatoirement avec la liste de bateaux [1,2,3]:", question4(genere_grille()[0]))


''' QUESTION 5: Donner un algorithme qui permet d’APPROXIMER le nombre total de grilles pour une
                liste de bateaux. Comparer les résultats avec le dénombrement des questions précédentes. 
                Est-ce une bonne manière de procéder pour la liste complète de bateaux ?
'''

'''
RÉPONSE:
    Comme cet algorithme renvoie une APPROXIMATION et non un nombre précis, nous nous appuierons sur le même modèle
    de calcul que dans la question 1 de la partie 2, ie en ne supprimant pas les cas où l'on a superposition de
    bateaux.
'''

def question5(liste_bateaux = [1,2,3,4,5]):
    """List[int]->int
    Renvoie l'approximation du nombre des cas possibles"""
    if(len(liste_bateaux)==0):
        return 0
    
    cpt = 1
    for bateau in liste_bateaux:
        cpt *= (DIMX + DIMY) * (DIMX - dic_bateaux[bateau] + 1)
        print(cpt)
    return cpt

'''QUESTION 6:Proposer des solutions pour approximer plus justement le nombre de configurations.
'''

'''REPONSE: Calculer le nombre de configuration des placements pour le bateau 1.
   Ensuite on va placer le bateau 1 aléatoirement. On fera les mêmes choses pour les 
   autres bateaux.
'''
def question6(combinaison,liste_bateaux,grille):
    if(len(liste_bateaux)==0):
        return combinaison
    cpt=0
    bateau=liste_bateaux[0]
    liste_peut_placer=[]
    for i in range(10):
        for j in range(10):
            if(peut_placer(grille, bateau,(i,j),'h')):
                cpt=cpt+1
                liste_peut_placer.append((i,j,'h'))
            if(peut_placer(grille, bateau,(i,j),'v')):
                cpt=cpt+1
                liste_peut_placer.append((i,j,'v'))
        
    combinaison=combinaison*cpt
    x,y,direction=rd.choice(liste_peut_placer)
    placer(grille, bateau,(x,y), direction)
    
    return question6(combinaison,liste_bateaux[1:],grille)

#cb=question6(1,[1,2,3,4,5],creer_grille())
#print(cb)

#print("\nQUESTION 5:")
#print("\t APPROXIMATION du nombre total de grilles pour une liste de bateaux [1,2,3,4,5]:", question5())

################################### PARTIE 3: MODÉLISATION PROBABILISTE DU JEU ###################################

class Bataille():
    def __init__(self,j1,j2):
        """Renvoie une instance de la classe de Bataile"""
        self.grillej1,self.listepositionj1=genere_grille()
        self.grillej2,self.listepositionj2=genere_grille()
        self.joueurs=[j1,j2]
        self.tirsucces=[False,False]#Si le joueur peut tirer un bateau ça devient true
        self.score=[0,0]
        self.courant=1 #Joueur 1 est pour 1,Joueur 2 est pour 2
        self.gagnant=0 #Si 0 aucun gagnat, 1 joueur 1 gagne 2 joueur 2 gagne
        
    def victoire(self):
        """void->int
        Renvoie le numéro de joueur gagnant"""
        winner=0 # 0 veut dire pas de gagnat  #1 veut dire joueur1 #2 veut dire joueur2
        #Conditions pour le joueur 1 gagne
        if(self.score[0]==MaxPointAPrendre()):
            winner=1
        if(self.score[1]==MaxPointAPrendre()):
            winner=2
        self.gagnant=winner
        return winner
    def joue(self):
        """void->void
        Choix de coup à joueur pour le joueur courant"""
        while(self.gagnant==0):
            if(self.courant==1):
                grilleadversaire=self.grillej2
            else:
                grilleadversaire=self.grillej1
            coup=self.joueurs[self.courant-1].joueCoup(grilleadversaire,self.tirsucces[self.courant-1])
            self.next(coup)
            self.gagnant=self.victoire()
    
    def next(self,coup):
        """(int,int)->void
        Change le bataille par rapport ausuccés du tir de joueur"""
        listeEnnemi=None
        if(self.courant==1):
            listeEnnemi=self.listepositionj2
        elif(self.courant==2):
            listeEnnemi=self.listepositionj1
        
        if coup in listeEnnemi:
            self.tirsucces[self.courant-1]=True
            self.score[self.courant-1]=self.score[self.courant-1]+1
        else:
            self.tirsucces[self.courant-1]=False
        
        self.changejoueur()
        
    def changejoueur(self):
        self.courant=3-self.courant
        
    def getGagnantJoueur(self):
        return self.joueurs[self.gagnant-1];
    def getGagnant(self):
        return self.gagnant
    
    def affiche(self):
        affiche(self.grillej1)
        affiche(self.grillej2)
    
    def reset(self):
        """void->void
        Regénere le bataille
        """
        self.grillej1,self.listepositionj1=genere_grille();
        self.grillej2,self.listepositionj2=genere_grille();
        self.joueurs=[self.j1,self.j2]        
        self.courant=1
        self.score=(0,0)

def distribution(joueur1,joueur2,nbrepete,showscore=True):
    scoretotal=[0,0]
    resultattotal=dict()
    i=0
    for k in range(1,101):
        resultattotal[k]=0
        
    for i in range(0,nbrepete):
        joueur1.reset()
        joueur2.reset()
        bataille=Bataille(joueur1,joueur2)
        bataille.joue()
        scoretotal[bataille.getGagnant()-1]=scoretotal[bataille.getGagnant()-1]+1
        nbCoupsDeGagnant=bataille.getGagnantJoueur().get_nb_coups()
        if nbCoupsDeGagnant in resultattotal.keys():
            resultattotal[nbCoupsDeGagnant]=resultattotal[nbCoupsDeGagnant]+1
            
    if(showscore):
        showScore(scoretotal,joueur1,joueur2)
        
        
    nbcoupsMoy=sum(key*resultattotal[key] for key in resultattotal.keys())/(1.0*nbrepete)
    
    return nbcoupsMoy,scoretotal

def showScore(score,j1,j2):
    """(int,int)->void
    Affiche le score total"""
    print("Le score est: {2} {0}-{1} {3}".format(score[0],score[1],j1.getName(),j2.getName()))

def Graphe(joueur1,joueur2,nomGraphe,partplay):
    listePartieJouee=[10*x for x in range(1,partplay)]
    listeNbCoupJouee=[]
    scoretot=[0,0]
    for partie in listePartieJouee:
        dist=distribution(joueur1,joueur2,partie,False)
        listeNbCoupJouee.append(dist[0])
        scoretot[0]=scoretot[0]+dist[1][0]
        scoretot[1]=scoretot[1]+dist[1][1]
     
    #print(sum(listeNbCoupJouee)/len(listeNbCoupJouee))    
    avg=round(sum(listeNbCoupJouee)/len(listeNbCoupJouee))
    print(avg)
    #Le score
    showScore(scoretot,joueur1,joueur2)
        
    plt.plot(listePartieJouee,listeNbCoupJouee)
    plt.xlabel("Le nombre de partie joué")
    plt.ylabel("Le nombre moyen de coup")
    #plt.yticks(list(range(avg-2,avg+3)))
    plt.title("Le nombre moyen des coup joués dans les parties")
    plt.savefig(nomGraphe)
    plt.clf()
    
    
#LA CLASSE MERE DES JOUEURS
class Joueur():
    def _init_(self):
        pass
    def joueCoup(self,grid):
        pass
    def get_nb_coups(self):
        pass
    def reset(self):
        pass
    def getName(self):
        pass

'''
QUESTION 1:
    
    En faisant des hypothèses que vous préciserez, quelle est l’espérance du nombre de coups joués avant de couler 
    tous les bateaux si on tire aléatoirement chaque coup? 
'''

'''
RÉPONSE:
    Soit X la variable aléatoire du nombre de coups qu'un joueur J doit jouer pour gagner.
    
    Pour connaître le nombre moyen de coups à jouer pour gagner, on fait un calcul 
    d'espérance.
    
    Formule de l'espérance:
        E(X) = somme des xi * P(X = xi)
    
    Ici: E(X) = somme pour i allant de 1 à 100 des i*P(X = i)
    
    ... avec pour tout i: P(X = i) = (16 parmi (i-1))/(17 parmi 100) 
        car la dernière case tirée en cas de victoire est forcément noire.
'''
    
def comb(n, k):
    """int,int->int
    Retourne Nombre de combinaisons de k parmi n"""
    
    if k > n//2:
        k = n-k
    x = 1
    y = 1
    i = n-k+1
    while i <= n:
        x = (x*i)//y
        y += 1
        i += 1
    return x

def esperance_theorique():
    e=0
    for i in range(17,101):
       e+=17*(comb(83,i-17)/comb(100,i))
    return e

print(esperance_theorique()) #Question1
#VERSION JOUEUR ALEATOIRE
class JoueurAlea(Joueur):
    def _init_(self):
        self.nbcoup=0
        self.listActionpossible=[(i,j) for i in range(DIMX) for j in range(DIMY)]
    def joueCoup(self,grid=None,tirsucces=False):
        coup=rd.choice(self.listActionpossible)
        self.listActionpossible.remove(coup)
        self.nbcoup=self.nbcoup+1
        return coup
    def get_nb_coups(self):
       return self.nbcoup
   
    def reset(self):
        self.nbcoup=0
        self.listActionpossible=[(i,j) for i in range(DIMX) for j in range(DIMY)]
    def getName(self):
        return "Joueur Aléatoire"
        
#VERSION JOUEUR HEURISTIQUE
voisinage=[(i,j) for i in range(-1,2) for j in range(-1,2) ]

class JoueurHeuri(Joueur):
    def __init__(self):
        self.nbcoup=0
        self.listActionpossible=[(i,j) for i in range(DIMX) for j in range(DIMY)]
        self.entour=[]
        self.last_coup=None
    def joueCoup(self,grid,tirsucces=False):
        if(tirsucces or self.entour !=[]):
            if(tirsucces):
                for v in voisinage:
                    v_x=self.last_coup[0]+v[0]
                    v_y=self.last_coup[1]+v[1]
                    v_prime=(v_x,v_y)
                    if(v_prime in self.listActionpossible):
                        if(grid[v_x,v_y]==grid[self.last_coup[0],self.last_coup[1]]):
                           self.entour.append(v_prime)
            if(self.entour ==[]):
                coup=rd.choice(self.listActionpossible)
            else:
                coup=rd.choice(self.entour)
                self.entour.remove(coup)
        else:
            coup=rd.choice(self.listActionpossible)
            
        if coup in self.listActionpossible:
            self.listActionpossible.remove(coup)
        self.nbcoup=self.nbcoup+1
        self.last_coup=coup
        return coup
    def get_nb_coups(self):
        return self.nbcoup
    def reset(self):
        self.nbcoup=0
        self.listActionpossible=[(i,j) for i in range(DIMX) for j in range(DIMY)]
        self.entour=[]
        self.last_coup=None
    def getName(self):
        return "Joueur Heuristique"
        


class JoueurProbaSimple(Joueur):
    
    def __init__(self):
        self.nbcoup = 0 # nombre de coups joués
        self.listActionpossible = [(i,j) for i in range(10) for j in range(10)] # liste des positions et directions pas encore testées par le joueur
        self.last_coup = None
        self.entour = [] #Liste des cases à tester en priorité (voisins des cases touchées)
        
        self.case_prio = (None, None, 0) # (id, x, y, proba) case prioritaire par sa probabilité élevée de contenir un bateau
        self.grid = np.zeros((DIMX, DIMY), dtype=int) # grille où le joueur stockera les cases adverses déjà touchées
        self.coup = None
        self.liste_bateaux = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} #liste du nombre de cases touchées par bateau
        
        self.bateauxRes = [1,2,3,4,5]
    
    def joueCoup(self, grid = None, touche=False):
        
        self.case_prio = (None, None, 0)
        
        if(touche==True or self.entour != []):
             
                if(touche==True):
                    
                    id = grid[self.last_coup[0], self.last_coup[1]]
                    self.grid[self.last_coup[0], self.last_coup[1]] = id
                    
                    if(id in self.liste_bateaux.keys()):
                            self.liste_bateaux[id] += 1
                            if self.liste_bateaux[id] == dic_bateaux[id]: self.bateauxRes.remove(id)
                    
                    for v in voisinage:
                        v_x = self.last_coup[0] + v[0] # abscisse du voisin considéré
                        v_y = self.last_coup[1] + v[1] # ordonnée du voisin considéré
                        
                        if (v_x, v_y) in self.listActionpossible and self.last_coup != None:
                            if grid[v_x,v_y] == grid[self.last_coup[0],self.last_coup[1]]:
                                self.entour.append((v_x,v_y))
            
                if self.entour == [] and self.listActionpossible!=[]: 
                    coup = rd.choice(self.listActionpossible)
                else:
                    if self.entour!=[]: 
                        coup = rd.choice(self.entour)
                    self.entour.remove(coup)
                    
                
                self.last_coup = self.coup

        else:       
            
            probas = np.zeros((DIMX, DIMY), dtype=float) # grille des probas pour chaque case de contenir un bateau
            nb_places_totales = 0
                
            for bateau in self.bateauxRes:
                #print("bateau: ", bateau)
        
                for i in range(DIMX):
                    for j in range(DIMY):
                        if peut_placer(self.grid, bateau, (i,j), 'h'):
                            for t in range(dic_bateaux[bateau]):
                                probas[i, j + t] += 1
                            nb_places_totales += 1
                        if peut_placer(self.grid, bateau, (i,j), 'v'):
                            for t in range(dic_bateaux[bateau]):
                                probas[i + t, j] += 1
                            nb_places_totales += 1
                
            for i in range(DIMX):
                for j in range(DIMY):
                    if(nb_places_totales!=0): proba = probas[i,j]/nb_places_totales
                    else: proba = 0
                    if proba > self.case_prio[2] and (i,j) in self.listActionpossible:
                        if self.listActionpossible!=[]: self.case_prio = (i, j, proba)
            #print("case_prio = ", self.case_prio)
            #print(probas)                
            # On a choisi la case prioritaire
            # On retourne le coup choisi:
            #print("bateau_prio = ", self.case_prio)
            x, y, proba = self.case_prio
            
            self.last_coup = (x, y)
            self.coup = (x,y)

        if(self.coup in self.listActionpossible): self.listActionpossible.remove(self.coup)
        self.nbcoup += 1
        
        
        return self.coup
    
    def get_nb_coups(self):
        return self.nbcoup
    
    def reset(self):
        self.nbcoup = 0 # nombre de coups joués
        self.listActionpossible = [(i,j) for i in range(10) for j in range(10)] # liste des positions et directions pas encore testées par le joueur
        self.last_coup = None
        self.case_prio = (None, None, 0) # (id, x, y, proba) case prioritaire par sa probabilité élevée de contenir un bateau
        self.grid = np.zeros((DIMX, DIMY), dtype=int) # grille où le joueur stockera les cases adverses déjà touchées
        self.coup = None
        self.liste_bateaux = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} #liste du nombre de cases touchées par bateau
        self.bateauxRes = [1,2,3,4,5]
    def getName(self):
        return "Joueur Probabiliste simplifié"
        




########################################## PARTIE 4: SENSEUR IMPARFAIT : À LA RECHERCHE DE L’USS SCORPION ##########################################

PS = 0.8 #probabilité que le senseur détecte l'objet s'il se trouve sur la case
COORD_CENTRE = ((3,6), (3,6))

def creer_grille_recherche():
    '''Crée une grille concue pour la recherche d'objets'''
    #Initialise grille avec probabilité πi pour chaque case
    grille = creer_grille_float()

    proba_centre_restante = 0.8
    proba_bords_restante = 0.2
    nb_cases_centre = 4 * 4
    for i in range(0, DIMX):
        for j in range(0, DIMY):
            if (i >= COORD_CENTRE[0][0] and i <= COORD_CENTRE[0][1] and j >= COORD_CENTRE[1][0] and j <= COORD_CENTRE[1][1]): #carré au centre de la grille
                # pi = rd.uniform(0, proba_centre_restante)
                # proba_centre_restante -= pi
                pi = proba_centre_restante / nb_cases_centre
            else:
                # pi = rd.uniform(0, proba_bords_restante)
                # proba_bords_restante -= pi
                pi = proba_bords_restante / (DIMX * DIMY - nb_cases_centre)
            grille[i,j] = pi

    return grille

def recherche_objet_perdu_naive():
    '''Recherche d'objet en itérant sur toutes les cases de la grille'''
    objet_trouve = False
    nb_essais = 1
    while (objet_trouve != True):
        for i in range(0, DIMX):
            for j in range(0, DIMY):
                coord_senseur = (i,j)
                if (coord_senseur == COORD_OBJ_PERDU and rd.random() < PS): #chance de détecter l'objet s'il se trouve sur la case
                    objet_trouve = True
                    break
                nb_essais += 1
            if (objet_trouve):
                break

       

    # print("recherche_objet_perdu_naive : Objet trouvé aux coordonnées " + str(coord_senseur) + " en " + str(nb_essais) + " essais")
    return nb_essais
    
def recherche_objet_perdu_random():
    '''Recherche d'objet en parcourant aléatoirement les cases de la grille'''
    objet_trouve = False
    nb_essais = 1
    while (objet_trouve != True):
        coord_senseur = (rd.randint(0,DIMX-1), rd.randint(0,DIMY-1))
        if (coord_senseur == COORD_OBJ_PERDU and rd.random() < PS): #chance de détecter l'objet s'il se trouve sur la case
            objet_trouve = True
            break

        nb_essais += 1

    # print("recherche_objet_perdu_random : Objet trouvé aux coordonnées " + str(coord_senseur) + " en " + str(nb_essais) + " essais")
    return nb_essais


def recherche_objet_perdu_proba():
    '''Recherche d'objet selon la probabilité de trouver l'objet dans chaque case'''
    grille = creer_grille_recherche()
    objet_trouve = False

    # print("Somme des probas = " + str(grille.sum()))

    nb_essais = 1
    while (objet_trouve != True):
        coord_max = np.unravel_index(grille.argmax(), grille.shape) #choisit la case avec la plus grande probabilité
        if (coord_max == COORD_OBJ_PERDU and rd.random() < PS): #chance de détecter l'objet s'il se trouve sur la case
            objet_trouve = True
            break

        # affiche_valeurs(grille, COORD_OBJ_PERDU, coord_max)

        #mise à jour de la proba de la case k ou l'on a pas détecté l'objet
        pi_k = grille[coord_max[0], coord_max[1]]
        pi_k_prime = (pi_k * (1 - PS)) / (1 - pi_k * PS)
        grille[coord_max[0], coord_max[1]] = pi_k_prime

        #mise à jour de la proba des autres cases
        p_Zk_zero = (1 - pi_k * PS) #v1
        # r = (1 - pi_k_prime) / (1 - pi_k) #v2
        for i in range(0, DIMX):
            for j in range(0, DIMY):
                if (i,j) != coord_max:
                    grille[i,j] = grille[i,j] / p_Zk_zero #v1
                    # grille[i,j] = grille[i,j] * r #v2

        nb_essais += 1

        # print("Somme des probas = " + str(grille.sum()))

    print("recherche_objet_perdu_proba : Objet trouvé aux coordonnées " + str(coord_max) + " en " + str(nb_essais) + " essais")
    return nb_essais


def tests_recherche_objet_perdu():

    nb_essais_naive = []
    nb_essais_random = []
    nb_essais_proba = []
    for i in range(0,1000):
        global COORD_OBJ_PERDU
        COORD_OBJ_PERDU = (rd.randint(COORD_CENTRE[0][0], COORD_CENTRE[0][1]), rd.randint(COORD_CENTRE[1][0], COORD_CENTRE[1][1]))
        print("Coordonnées objet perdu = " + str(COORD_OBJ_PERDU))
        nb_essais_naive.append(recherche_objet_perdu_naive())
        nb_essais_random.append(recherche_objet_perdu_random())
        nb_essais_proba.append(recherche_objet_perdu_proba())

    # print(nb_essais_naive)
    # print(nb_essais_random)
    # print(nb_essais_proba)

    print(np.mean(nb_essais_naive))
    print(np.std(nb_essais_naive))
    print(np.mean(nb_essais_random))
    print(np.std(nb_essais_random))
    print(np.mean(nb_essais_proba))
    print(np.std(nb_essais_proba))

    bins = np.linspace(0, min(400, max(max(nb_essais_naive), max(nb_essais_random), max(nb_essais_proba))), 200)

    plt.hist(nb_essais_naive, bins, alpha=0.5, label='naive', edgecolor="k")
    plt.hist(nb_essais_random,bins, alpha=0.5, label='random', edgecolor="k")
    plt.hist(nb_essais_proba, bins, alpha=0.5, label='proba', edgecolor="k")
    plt.legend(loc='upper right')
    plt.show()




def main():
    rd.seed()    
    joueur1=JoueurAlea()
    joueur2=JoueurAlea()
    joueur3=JoueurHeuri()
    joueur4=JoueurAlea()
    joueur5=JoueurProbaSimple()
    #Graphe(joueur1,joueur2,"AleavsAlea",101)
    Graphe(joueur3,joueur4,"HeurivsAlea",101)
    #Graphe(joueur5,joueur3,"ProbaSimplevsHeuri",51)

    #tests_recherche_objet_perdu()
    
main()