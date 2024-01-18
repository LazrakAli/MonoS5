import sys
from plateau import Plateau
from joueur import Joueur
from joueuralea import JoueurAlea
from monte_carlo import Monte_Carlo
from partie import Partie
import matplotlib.pyplot
import ast
import numpy as np
import math
from joueuruct import JoueurUCT
def recuperation_donnee(nom):
    #stock le nombre de coup par victoire de chaque joueur
    coup_victoire1 = []
    coup_victoire2= []

    with open(f'{nom}.txt', 'r') as fichier:
        for ligne in fichier:
            a=ligne.split(',')
            x,y=map(int,a)
            if y==1:
                coup_victoire1.append(x)
            if y==-1:
                coup_victoire2.append(x)

    # Afficher les résultats
    return (coup_victoire1,coup_victoire2)

def plot_histogram(joueur1, joueur2):
    matplotlib.pyplot.hist([joueur1, joueur2], bins=range(1, max(max(joueur1), max(joueur2)) + 2), alpha=0.5, label=['Joueur 1', 'Joueur 2'], color=['blue', 'yellow'])
    matplotlib.pyplot.xlabel('Nombre de coups')
    matplotlib.pyplot.ylabel('Fréquence')
    matplotlib.pyplot.legend(loc='upper right')
    matplotlib.pyplot.title('Histogramme des coups des joueurs par victoire')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.show()

def moyenne_nbcoups(nom):
    a,b=recuperation_donnee(nom)
    n1=0 #somme des nombres de coups du joueur 1 
    n2=0 #somme des nombres de coups du joueur 2

    for i in a:
        n1+=i
    
    for i in b:
        n2+=i
    
    return(n1/len(a),n2/len(b))

def mediane_nbcoups(nom):
    a, b = recuperation_donnee(nom)
    med1 = 0
    med2 = 0
    atrie = sorted(a)
    btrie = sorted(b)

    if len(a) % 2 == 1:
        med1 = atrie[len(a) // 2]
    else:
        med1 = (atrie[len(a) // 2 - 1] + atrie[len(a) // 2]) / 2

    if len(b) % 2 == 1:
        med2 = btrie[len(b) // 2]
    else:
        med2 = (btrie[len(b) // 2 - 1] + btrie[len(b) // 2]) / 2

    return med1, med2


#nom=fichier de resultat
def variance_nbcoups(nom):

    m1,m2=moyenne_nbcoups(nom)
    sum1=0
    sum2=0
    a,b=recuperation_donnee(nom)

    for i in a:
        sum1+=(i-m1)**2
    for i in b:
        sum2+=(i-m2)**2
    return((1/(len(a)-1)*sum1),(1/(len(b)-1))*sum2)


#nom=fichier de resultat
def ecart_type_nbcoups(nom):
    s1,s2=variance_nbcoups(nom)
    return (math.sqrt(s1),math.sqrt(s2))


def stat(nom,jeu):
    a,b=recuperation_donnee(nom)
    plot_histogram(a,b)
    m1,m2=moyenne_nbcoups(nom)
    print(f"Pour Plateau {jeu.plateau.L}x{jeu.plateau.C}")
    print("moyennes :\n",m1,m2)
    print("médianes:\n",mediane_nbcoups(nom))
    print("variance :\n",variance_nbcoups(nom))
    print("ecart type :\n",ecart_type_nbcoups(nom))


partie=Partie(Joueur(1,nom='Ali'),Joueur(-1,nom='Hichem Amer'))
partie.plateau.run(partie.joueur1,partie.joueur2,affichage_complet=True)
#partie.ecriture_dans_un_fichier('test')
#stat('test',partie)

