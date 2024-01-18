# Ali Lazrak
# Samuel Ktorza

from statistics import mean
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib.pyplot as plt
import math
from utils import AbstractClassifier, drawGraph
import scipy


def getPrior(data):
    """
        calcule la probabilité a priori de la classe 1 ainsi que l'intervalle de confiance à 95% pour l'estimation de cette probabilité
        @param: dataframe a étudie
        @return : dictionnaire contenant 3 clés 'estimation', 'min5pourcent', 'max5pourcent'
    """
    values = data['target'].tolist()
    moy = np.mean(values)
    var = moy*(1-moy)
    ett = math.sqrt(var) #ecart-type
    #intervalle de conf 95%
    min5percent = moy - 1.96 * ett / math.sqrt(len(values))
    max5percent = moy + 1.96 * ett / math.sqrt(len(values))
    return {'estimation' : moy, 'min5pourcent' : min5percent, 'max5percent' : max5percent}






class APrioriClassifier(AbstractClassifier):
    """
        estime la classe de chaque individu par la classe majoritaire.
    """

    def __init__(self):
        """
            constructeur de APrioriClassifier
        """
        self.class_Maj = 1 

    def estimClass(self,attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        @param: attr: Dict[str,val] (Le dictionnaire nom-valeur des attributs)
        @returns:la classe 0 ou 1 estimée
        """
        return self.class_Maj
    


    def statsOnDF(self, df):
        '''
        Calcule la précision et le rappel du classifieur
        
        @return: le dictionnaire des statistiques de dictionnaire df
        '''
        vp, vn, fp, fn = 0, 0, 0, 0
        precision = 0
        rappel = 0
        for t in df.itertuples():
            dic=t._asdict()
            dic.pop('Index',None)
            if dic['target'] ==1 :
                if self.estimClass(dic) == 1:
                    vp = vp+1
                else:
                    fn = fn+1
            else:
                if self.estimClass(dic) == 1:
                    fp = fp+1
                else:
                    vn = vn+1
        precision = vp/(vp+fp)
        rappel = vp/(fn+vp)
        return {'VP':vp,'VN':vn,'FP':fp,'FN':fn,'Précision':precision,'Rappel':rappel}






def P2D_l(df, attr):
    '''
    Calcule dans la dataframe la probabilité P(attr|target) sous la forme
    d'un dictionnaire associant à la valeur 't' un dictionnaire associant à la
    valeur 'a' la probabilité P(attr = a | target = t)

    @param: df (pd.DataFrame): Le dataframe contenant les données.
            attr (str): Le nom de la colonne d'attributs pour laquelle nous calculons les probabilités conditionnelles.
    @return: dict: Un dictionnaire représentant les probabilités conditionnelles P(attr|target).
    '''
    dict_ret = dict()
    possible_target_values = df.target.unique()  #liste des valeurs uniques dans la colonne cible
    for target_val in possible_target_values:
        df_target = df[df.target == target_val]
        nb_same_target = df_target.shape[0]
        dict_ret[target_val] = dict()  # dico pour stocker les probabilités conditionnelles par valeur de 'a' pour chaque valeur de 't'

        possible_attr_values = df[attr].unique()  #liste des valeurs uniques dans la colonne d'attributs
        for attr_val in possible_attr_values:
            df_same_attr = df_target[df_target[attr] == attr_val]
            nb_same_attr_val = df_same_attr.shape[0]
            proba = nb_same_attr_val / nb_same_target
            dict_ret[target_val][attr_val] = proba  #Stocke la probabilité P(attr = a | target = t)

    return dict_ret



def P2D_p(df, attr):
    '''
    Calcule dans le dataframe la probabilité P(target|attr) sous la forme
    d'un dictionnaire associant à la valeur t un dictionnaire associant à la
    valeur 't' la probabilité P(target = t|attr = a)

    @param:
        df (pd.DataFrame): Le dataframe contenant les données.
        attr (str): Le nom de la colonne d'attributs pour laquelle nous calculons les probabilités conditionnelles.

    @return:
        dict: Un dictionnaire représentant les probabilités conditionnelles P(target|attr).
    '''
    dict_ret = dict()
    possible_attr_values = df[attr].unique()  #liste des valeurs uniques dans la colonne d'attributs
    for attr_val in possible_attr_values:
        df_same_attr = df[df[attr] == attr_val]
        nb_same_attr_val = df_same_attr.shape[0]
        dict_ret[attr_val] = dict()  #dico pour stocker les probabilités conditionnelles par valeur de 't' pour chaque valeur de 'a'

        possible_target_values = df.target.unique()  #liste des valeurs uniques dans la colonne cible
        for target_val in possible_target_values:
            df_target = df_same_attr[df_same_attr.target == target_val]
            nb_same_target = df_target.shape[0]
            proba = nb_same_target / nb_same_attr_val
            dict_ret[attr_val][target_val] = proba  #stocke la probabilité P(target = t | attr = a)

    return dict_ret






class ML2DClassifier(APrioriClassifier):
    '''
    Classifie selon le maximum de vraisemblance
    '''

    def __init__(self, attrs, attr):
        self.probas = P2D_l(attrs, attr)
        self.class_Maj = attr

    def estimClass(self, attrs):
        """
        À partir d'un dictionnaire d'attributs, estime la classe 0 ou 1.

        @param: attrs (dict): Le dictionnaire nom-valeur des attributs.
        @return: int: La classe 0 ou 1 estimée.
        """
        if self.probas[0][attrs[self.class_Maj]] > self.probas[1][attrs[self.class_Maj]]:
            return 0
        return 1





class MAP2DClassifier(APrioriClassifier):
    '''
    Classifie selon le maximum a posteriori (MAP)
    '''

    def __init__(self, attrs, attr):
        '''
        Initialise un nouvel objet MAP2DClassifier.

        @param:
            attrs (pd.DataFrame): Le dataframe contenant les données.
            attr (str): Le nom de la colonne d'attributs pour laquelle nous calculons les probabilités conditionnelles.
        '''
        self.probas = P2D_p(attrs, attr)  # Calcul de la table P2Dp
        self.attr = attr  # Stockage du nom de la colonne d'attributs

    def estimClass(self, attrs):
        '''
        À partir d'un dictionnaire d'attributs, estime la classe 0 ou 1 en utilisant le maximum a posteriori (MAP).

        @param: attrs (dict): Le dictionnaire nom-valeur des attributs.
        @return: int: La classe 0 ou 1 estimée.
        '''
        # Comparaison des probabilités a posteriori pour choisir la classe
        if self.probas[attrs[self.attr]][0] > self.probas[attrs[self.attr]][1]:
            return 0
        return 1





#####
# Question 2.4 : comparaison
#####
#APrioriClassifier affiche une précision de 0.69 sur le jeu de test, 
#nettement inférieure à ML2DClassifier et MAP2DClassifier qui obtiennent respectivement 0.89 et 0.86 de précision (arrondies). 
#Bien que ces deux derniers résultats soient très proches, le ML2DClassifier présente un léger avantage. 
#Les rappels de ML2DClassifier et MAP2DClassifier sont quasiment identiques, avec des valeurs de 0.82 et 0.83 respectivement. 
#Un point notable concerne le APrioriClassifier, qui affiche un rappel de 1, expliqué par l'absence de vrais négatifs ou de faux négatifs. 
#Ainsi, malgré sa faible précision, APrioriClassifier conserve tous les éléments pertinents de la base
#####




def nbParams(df, attrs=None):
    '''
    Calcule la taille mémoire des tables P(target|attr1,..,attrk) étant donné un dataframe 
    et la liste [target,attr1,...,attrl] en supposant qu'un float est représenté sur 8 octets.
    Le résultat est directement affiché et non retourné.

    @param df: le dataframe contenant les données de la base
    @param attrs: la liste des attributs à considérer dans le calcul
    '''
    
    # Si on n'indique pas le nom des attributs, on calcule pour tous les attributs
    if attrs is None:
        attrs = [str(key) for key in df.keys()]

    cpt = 1
    for attr in attrs:
        # dic est un dictionnaire qui contient les attributs souhaités
        dic = {a: None for a in df[attr]}
        cpt *= len(dic)

    # Affichage:
    nb_octets = cpt * 8
    aff = ""
    aff = aff + str(len(attrs)) + " variable(s) : " + str(nb_octets) + " octets"

    s_o = ""
    s_ko = ""
    s_mo = ""
    s_go = ""

    if nb_octets >= 1024:
        aff = aff + " = "
        nb_ko = nb_octets // 1024
        nb_octets = nb_octets % 1024
        s_o = str(nb_octets) + "o"
        s_ko = str(nb_ko) + "ko "

        if nb_ko >= 1024:
            nb_mo = nb_ko // 1024
            nb_ko = nb_ko % 1024
            s_ko = str(nb_ko) + "ko "
            s_mo = str(nb_mo) + "mo "

            if nb_mo >= 1024:
                nb_go = nb_mo // 1024
                nb_mo = nb_mo % 1024
                s_mo = str(nb_mo) + "mo "
                s_go = str(nb_go) + "go "

    aff = aff + s_go + s_mo + s_ko + s_o
    print(aff)



def nbParamsIndep(df, attrs=None):
    """
    Calcule la taille mémoire nécessaire pour représenter les tables de probabilité 
    étant donné un dataframe, en supposant qu'un float est représenté sur 8 octets et en supposant 
    l'indépendance des variables.
    Le résultat est directement affiché et non retourné.
    
    @param df: le dataframe contenant les données de la base
    @param attrs: la liste des attributs à considérer dans le calcul
    """
    # Si on n'indique pas le nom des attributs, on calcule pour tous les attributs
    if attrs is None:
        attrs = [str(key) for key in df.keys()]

    cpt = 0
    for attr in attrs:
        # dic est un dictionnaire qui contient des attributs souhaités
        dic = {a: None for a in df[attr]}
        cpt += len(dic)

    # Affichage:
    nb_octets = cpt * 8
    aff = ""
    aff = aff + str(len(attrs)) + " variable(s) : " + str(nb_octets) + " octets"

    s_o = ""
    s_ko = ""
    s_mo = ""
    s_go = ""
    if nb_octets >= 1024:
        aff = aff + " = "
        nb_ko = nb_octets // 1024
        nb_octets = nb_octets % 1024
        s_o = str(nb_octets) + "o"
        s_ko = str(nb_ko) + "ko "

        if nb_ko >= 1024:
            nb_mo = nb_ko // 1024
            nb_ko = nb_ko % 1024
            s_ko = str(nb_ko) + "ko "
            s_mo = str(nb_mo) + "mo "
            if nb_mo >= 1024:
                nb_go = nb_mo // 1024
                nb_mo = nb_mo % 1024
                s_mo = str(nb_mo) + "mo "
                s_go = str(nb_go) + "go "

    aff = aff + s_go + s_mo + s_ko + s_o
    print(aff)



#####
# Question 3.3.a : preuve
#####
# définition d'indépendance conditionnelle. Deux événements A et B sont indépendants sachant un troisième événement C
# si P(A∣B,C) = P(A∣C) 
# P(A,B,C) = P(A∣B,C)⋅P(B∣C)⋅P(C)
# P(A,B,C) = P(A∣C)⋅P(B∣C)⋅P(C)
# Par identification avec P(A,B,C)=P(A)⋅P(B∣A)⋅P(C∣B)
#            P(A∣C)=P(A)   
#            P(B∣C)=P(B∣A)
#            P(C) =P(C|B)
# Ainsi nous pouvons représenter la loi jointe sous la forme d'indépendances conditionnelles.
#####



#####
# Question 3.3.b : complexité en indépendance partielle
#####
#Sans l'indépendance conditionnelle : 
#    5^3*8=1000 octets.
#Avec l'indépendance conditionnelle: 
#    (2*5^2+5)*8=440$ octets
#####



#####
# Question 4.1 : Exemples
#####
# Variables complètement indépendantes
# utils.drawGraphHorizontal("A;B;C;D;E")
# Variables sans aucune indépendance
#utils.drawGraphHorizontal("A->B;A->C;A->D;A->E;B->C;B->D;B->E;C->D;C->E;D->E")
#####

#####
#Question 4.2 : naïve Bayes
#####
# P(attr1,attr2,attr3,...|target) se decompose de la maniere suivante:
#       P(attr1,attr2,attr3,…∣target)=P(attr1∣target)*P(attr2∣target)*P(attr3 ∣target)*...
# P(target|attr1,attr2,attr3...)= K * P(target)* P(attr1|target)*P(attr2|target)*...
# K=le dénominateur dans le théorème de Bayes, elle peut etre negliger car elle n'affecte pas la comparaison des probabilités.
#####



def drawNaiveBayes(df, class_attr):
    '''
    Dessine un graphe représentant un modèle naïve bayes à partir d'un dataframe et du nom de l'attribut qui est la classe.

    :param1 df: pandas.DataFrame (Le dataframe contenant les données de la base)
    @param2 class_attr: str (Le nom de l'attribut-classe)
    @return: L'image représentant le graphe du modèle naïve bayes.
    '''
    #chaîne vide pour stocker la représentation du modèle naïve bayes
    modelStr = ""

    #parcourt les colonnes du df
    for attr in df.columns:
        #Si l'attribut n'est pas l'attribut de classe, crée un arc dans le modèle naïve bayes
        if attr != class_attr:
            modelStr += class_attr + "->" + attr + ";"

    #utilise la fonction drawGraph pour dessiner le graphe correspondant
    return drawGraph(modelStr)



def nbParamsNaiveBayes(df, class_attr, attrs=None):
    """
    Calcule la taille mémoire pour représenter les tables de probabilité en utilisant l'hypothèse du Naive Bayes.
    Le résultat est directement affiché et non retourné.
    
    @param1 df: Le dataframe contenant les données de la base.
    @param2 class_attr: Le nom de l'attribut-classe.
    @param3 attrs: La liste des attributs à considérer dans le calcul. Si non précisé, tous les attributs du dataframe sont pris en compte.
    """
    # Si la liste d'attributs n'est pas précisée, on prend tous les attributs du df
    if attrs is None:
        attrs = df.columns

    # Nombre de valeurs uniques de l'attribut classe
    nb_possible_values_class_attr = len(df[class_attr].unique())

    # Initialisation du nombre de valeurs des attributs
    nb_values = 0

    # Itération sur tous les attributs pour calculer le nombre total de valeurs possibles
    for attr in attrs:
        if attr != class_attr:
            # Nombre de valeurs uniques de l'attribut courant
            nb_possible_values = len(df[attr].unique())
            # Accumulation du nombre total de valeurs
            nb_values += nb_possible_values * nb_possible_values_class_attr

    # Calcul de la taille en octets
    nb_octets = (nb_possible_values_class_attr + nb_values) * 8

    # La valeur à retourner
    val_retour = nb_octets

    # Construction de la chaîne d'affichage
    aff = str(len(attrs)) + " variable(s) : " + str(nb_octets) + " octets"

    # Gestion des unités (octets, Ko, Mo, Go)
    s_o = ""
    s_ko = ""
    s_mo = ""
    s_go = ""
    if nb_octets >= 1024:
        aff = aff + " = "
        nb_ko = nb_octets // 1024
        nb_octets = nb_octets % 1024
        s_o = str(nb_octets) + "o"
        s_ko = str(nb_ko) + "ko "
        if nb_ko >= 1024:
            nb_mo = nb_ko // 1024
            nb_ko = nb_ko % 1024
            s_ko = str(nb_ko) + "ko "
            s_mo = str(nb_mo) + "mo "
            if nb_mo >= 1024:
                nb_go = nb_mo // 1024
                nb_mo = nb_mo % 1024
                s_mo = str(nb_mo) + "mo "
                s_go = str(nb_go) + "go "

    # Finalisation de la chaîne d'affichage
    aff = aff + s_go + s_mo + s_ko + s_o

    # Affichage du résultat
    print(aff)

    # Retour de la valeur calculée
    return val_retour





class MLNaiveBayesClassifier(APrioriClassifier):
    '''
    Estime la classe d'un individu en utilisant le maximum de vraisemblance et l'hypothèse du Naïve Bayes
    '''
    def __init__(self, attrs):
        self.probas = dict() #clé=attr, valeur=P(attr|target)
        for attr in attrs :
            if attr != "target" :
                self.probas[attr] = P2D_l(attrs, attr)
        
    def estimProbas(self, attrs):
        """
		à partir d'un dictionnaire d'attributs, estime la probabilité que la classe soit 0 ou 1
		:param attrs: le  dictionnaire nom-valeur des attributs
		:return: un dictionnaire avec la probabilité que la classe soit 0, et celle que la classe soit 1
		"""
        proba_class_is_0 = 1 #probabilité que la classe (target) soit 0
        proba_class_is_1 = 1 #probabilité que la classe (target) soit 1
        for attr in attrs.keys() :
            if attr != "target" :
                try:
                    proba_class_is_0 *= self.probas[attr][0][attrs[attr]]
                except KeyError :
                    proba_class_is_0 *= 0 #hypothèse : si on ne trouve pas la valeur dans la base d'entrainement, la probabilité d'avoir cette valeur est 0
                try:
                    proba_class_is_1 *= self.probas[attr][1][attrs[attr]]
                except KeyError :
                    proba_class_is_1 *= 0
        
        return {0 : proba_class_is_0, 1 : proba_class_is_1}

    def estimClass(self, attrs):
        """
		à partir d'un dictionnaire d'attributs, estime la classe 0 ou 1
		:param attrs: le  dictionnaire nom-valeur des attributs
		:return: la classe 0 ou 1 estimée
		"""
        proba_class = self.estimProbas(attrs)
        return max(proba_class, key=proba_class.get)




class MAPNaiveBayesClassifier(APrioriClassifier):
    '''
    Estime la classe d'un individu en utilisant le maximum a posteriori et l'hypothèse du Naïve Bayes
    '''
    def __init__(self, attrs):
        # Initialisation des probabilités conditionnelles P(attr|target) pour chaque attribut
        self.probas = dict() # Clé=attr, valeur=P(attr|target)
        for attr in attrs:
            if attr != "target":
                self.probas[attr] = P2D_l(attrs, attr)

        # Calcul des probabilités a priori P(target=1) et P(target=0)
        prior_estimation = getPrior(attrs)
        self.p_target_1 = prior_estimation['estimation'] # P(target = 1)
        self.p_target_0 = 1 - self.p_target_1

    def estimProbas(self, attrs):
        """
        À partir d'un dictionnaire d'attributs, estime la probabilité que la classe soit 0 ou 1.
        :param attrs: Le dictionnaire nom-valeur des attributs.
        :return: Un dictionnaire avec la probabilité que la classe soit 0, et celle que la classe soit 1.
        """
        proba_class_is_0 = 1 # Probabilité que la classe (target) soit 0
        proba_class_is_1 = 1 # Probabilité que la classe (target) soit 1
        
        for attr in attrs.keys():
            if attr != "target":
                try:
                    proba_class_is_0 *= self.probas[attr][0][attrs[attr]]
                except KeyError:
                    proba_class_is_0 *= 0 # Hypothèse : si on ne trouve pas la valeur dans la base d'entrainement, la probabilité d'avoir cette valeur est 0
                try:
                    proba_class_is_1 *= self.probas[attr][1][attrs[attr]]
                except KeyError:
                    proba_class_is_1 *= 0

        # P(attr1, attr2, ..., attrn) = P(attr1|target=0)...P(attrn|target=0)P(target=0) + P(attr1|target=1)...P(attrn|target=1)P(target=1)
        if proba_class_is_0 == 0 and proba_class_is_1 == 0: # Si les deux probabilités sont égales, on choisit toujours la classe 0
            proba_class_is_0 = 1
        p_attrs = (proba_class_is_0 * self.p_target_0 + proba_class_is_1 * self.p_target_1)

        # Formule Naive Bayes a posteriori
        return {0: proba_class_is_0 * self.p_target_0 / p_attrs, 1: proba_class_is_1 * self.p_target_1 / p_attrs}

    def estimClass(self, attrs):
        """
        À partir d'un dictionnaire d'attributs, estime la classe 0 ou 1.
        :param attrs: Le dictionnaire nom-valeur des attributs.
        :return: La classe 0 ou 1 estimée.
        """
        proba_class = self.estimProbas(attrs)
        return max(proba_class, key=proba_class.get)



def isIndepFromTarget(df, attr, x):
    """
    Vérifie si l'attribut attr est indépendant de target au seuil de x% en utilisant le test du chi2.
    
    @param1 df: Le df 
    @param attr: Le nom de l'attribut à vérifier
    @param x: Le seuil d'indépendance en pourcentage
    @return: True si attr est indépendant de target, False sinon
    """
    # Crée une table de contingence en croisant les fréquences entre la cible ('target') et l'attribut spécifié (attr)
    contingency_tables = pd.crosstab(df['target'], df[attr])

    # Effectue le test du chi2 sur la table de contingence
    # Retourne le chi2 (statistique du test), p (niveau de signification), dof (degrés de liberté), ex (tableau des fréquences attendues)
    chi2, p, dof, ex =scipy.stats.chi2_contingency(contingency_tables)

    # Renvoie True si la p-valeur est supérieure ou égale au seuil x, indiquant que l'indépendance entre la cible et l'attribut n'est pas rejetée
    return p>=x


class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    """
    Utilise le maximum de vraisemblance pour estimer la classe d'un individu en
    utilisant l'hypothèse du Naïve Bayes.
    Les attributs indépendants de target ne sont pas considérés.
    """
    def __init__(self, df, x):
        # Initialise un dictionnaire indiquant si les attributs sont indépendants de target
        self.independence = dict()

        # Initialise un dictionnaire de probabilités conditionnelles P(attr|target)
        self.P2DL_dic = dict()

        # Parcourt les attributs de la base de données
        for attr in df:
            if attr != "target":
                # Vérifie l'indépendance des variables par rapport à target
                self.independence[attr] = isIndepFromTarget(df, attr, x)

                # Si l'attribut n'est pas indépendant, calcule les probabilités conditionnelles
                if not self.independence[attr]:
                    self.P2DL_dic[attr] = P2D_l(df, attr)

        # Initialise un dictionnaire pour stocker les probabilités de classe
        self.dic = {0: None, 1: None}

    def draw(self):
        '''
        Dessine un graphe représentant le modèle naïve bayes sans les attributs indépendants de target
        '''
        # Convertit le dictionnaire de probabilités conditionnelles en DataFrame pour dessiner le graphe
        attrs = pd.DataFrame.from_dict(self.P2DL_dic)
        return drawNaiveBayes(attrs, "target")

    def estimProbas(self, attrs):
        """
		À partir d'un dictionnaire d'attributs, estime la probabilité que la classe soit 0 ou 1.
		:param attrs: Le dictionnaire nom-valeur des attributs.
		:return: Un dictionnaire avec la probabilité que la classe soit 0, et celle que la classe soit 1.
		"""
        # Initialise les probabilités de classe
        attr_0 = 1
        attr_1 = 1

        # Parcourt les attributs du dictionnaire d'attributs
        for attr in attrs.keys():
            # Vérifie si l'attribut est indépendant et s'il n'est pas la cible
            if attr != "target" and not self.independence[attr]:
                # Si la valeur de l'attribut n'est pas présente dans la base d'entraînement, retourne des probabilités nulles
                if attrs[attr] not in self.P2DL_dic[attr][0].keys():
                    self.dic[0] = 0
                    self.dic[1] = 0
                    return self.dic

                # Récupère les probabilités conditionnelles P(attr|target=0) et P(attr|target=1)
                p0 = self.P2DL_dic[attr][0][attrs[attr]]
                p1 = self.P2DL_dic[attr][1][attrs[attr]]

                # Met à jour les probabilités de classe
                attr_0 *= p0
                attr_1 *= p1

        # Calcule la somme des probabilités pour normaliser
        somme = attr_0 + attr_1

        # Si la somme est nulle, retourne des probabilités nulles
        if somme == 0:
            self.dic[0] = 0
            self.dic[1] = 0
            return self.dic

        # Calcule les probabilités normalisées de classe
        self.dic[0] = attr_0 / somme
        self.dic[1] = attr_1 / somme
        return self.dic




class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    """
    Utilise le maximum de vraisemblance pour estimer la classe d'un individu en
    utilisant l'hypothèse du Naïve Bayes.
    Les attributs indépendants de target ne sont pas considérés.
    """
    def __init__(self, df, x):
        self.independence = dict() #dict associant à chaque attr un bool indiquant s'il est indépendant ou non de target

        self.P2DL_dic = dict() #clé=attr, valeur=P(attr|target)
        for attr in df :
            if attr != "target" :
                #Vérifie l'indépendance des variables par rapport à target
                self.independence[attr] = isIndepFromTarget(df, attr, x)
                if self.independence[attr] == False:
                    #Calcul des probabilités
                    self.P2DL_dic[attr] = P2D_l(df, attr)

        self.dic = {0: None, 1: None}
        #Calcul P(target)
        self.p_t0 = df[(df['target'] == 0)].shape[0] / len(df) #P(target=0)
        self.p_t1 = df[(df['target'] == 1)].shape[0] / len(df) #P(target=1)

    def draw(self):
        '''
        Dessine un graphe représentant le modèle naïve bayes sans les attributs indépendants de target
        '''
        attrs = pd.DataFrame.from_dict(self.P2DL_dic)
        return drawNaiveBayes(attrs, "target")
    
    def estimProbas(self, attrs):
        """
        à partir d'un dictionnaire d'attributs, estime la probabilité que la classe soit 0 ou 1
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: un dictionnaire avec la probabilité que la classe soit 0, et celle que la classe soit 1
        """
        attr_0=1
        attr_1=1

        for attr in attrs.keys():
            if attr != "target" and self.independence[attr] == False :
                if attrs[attr] not in self.P2DL_dic[attr][0].keys():
                    self.dic[0] = 0
                    self.dic[1] = 0
                    return self.dic
                
                p0 = self.P2DL_dic[attr][0][attrs[attr]] #P(attr|target=0)
                p1 = self.P2DL_dic[attr][1][attrs[attr]] #P(attr|target=1)

                attr_0 = attr_0 * p0
                attr_1 = attr_1 * p1
            
        somme = attr_0 * self.p_t0 + attr_1 * self.p_t1

        if somme == 0:
            self.dic[0] = 0
            self.dic[1] = 0
            return self.dic

        self.dic[0] = attr_0 * self.p_t0 / somme
        self.dic[1] = attr_1 * self.p_t1 / somme
        return self.dic

    
def MeanForSymetricWeights(matrice):
    
    mean=0
    for i in range(0,len(matrice)):
        for j in range(0,len(matrice[0])):
            mean+=matrice[i][j]
    mean/=(len(matrice)*len(matrice[0])-len(matrice))  #on divise pas le nombre d'éléments - la diagonale
    return mean


#####
#Question 6.1

#####
#Le point idéal dans la représentation graphique des points (précision, rappel) est le coin supérieur droit, 
#où la précision et le rappel sont tous deux égaux à 1. Cela signifie que le classifieur a une précision maximale.

#a travers une Courbe (Precision-Recall) cette courbe représente la précision par rapport au rappel à 
#différents seuils de probabilité. 
#Un classifieur idéal aurait une courbe PR se rapprochant du coin supérieur droit.
#####


def mapClassifiers(dic, df):
    '''
    À partir d'un dictionnaire dic de classifiers et d'un dataframe df, représente graphiquement
    les classifiers dans l'espace (précision, rappel)
    @param1 dic: dictionnaire de classifieurs
    @param2 df: le dataframe contenant les données de la base
    '''
    plt.figure(figsize=(7, 5))
  
    for key in dic:
        # Obtient les statistiques (précision et rappel) du classifieur sur le dataframe
        cl = dic[key].statsOnDF(df)
        precision = cl['Précision']
        rappel = cl['Rappel']
        
        # Affiche le nom du classifieur à la position (précision, rappel) avec un marqueur 'x' de couleur rouge
        plt.scatter(precision, rappel, marker='x', color='r')
        plt.annotate(key, (precision, rappel))


#####
#Question 6.3 : Conclusion
#####
#Pour comparer les performances des classificateurs, le graphique à observer est le graphique lié à l'ensemble de tests, 
#qui représente la capacité du classificateur à estimer les classes d'individus qu'il n'a pas vus lors de l'entraînement.
#Nous voyons ici que les classificateurs 4, 5, 6 et 7 se trouvent tous dans le coin inférieur droit du graphique, 
#avec une grande précision mais un faible rappel.
#En revanche, le classificateur 1 a un rappel élevé mais une faible précision, 
#tandis que les classificateurs 2 et 3 ont un rappel et un rappel plus cohérents.
#####





def MutualInformation(df, X, Y):
    """
    Calcule l'information mutuelle conditionnelle entre les variables X et Y, sachant Z, dans un dataframe.

    @param1 df: DataFrame contenant les données.
    @param2 X: Nom de la première variable.
    @param3 Y: Nom de la deuxième variable.
    @param4 Z: Nom de la variable conditionnelle.

    @return: Valeur de l'information mutuelle conditionnelle entre X et Y sachant Z.
    """
    # Calcul des probabilités conditionnelles P(Y|X) à partir du dataframe
    proba_table = df.groupby(X)[Y].value_counts() / df.groupby(X)[Y].count()

    # Initialisation de l'information mutuelle
    information = 0.0

    # Récupération des indices de la table de probabilités
    list_values_index = proba_table.index.values.tolist()

    # Initialisation d'un dictionnaire pour stocker les valeurs uniques de Y pour chaque valeur de X
    dict_key_unique_values = {}

    # Remplissage du dictionnaire avec les valeurs uniques de Y pour chaque valeur de X
    for x in df[X].unique():
        dict_key_unique_values[x] = []

    for (x, y) in list_values_index:
        dict_key_unique_values[x].append(y)

    # Calcul de l'information mutuelle
    for x in df[X].unique():
        # Calcul de la probabilité marginale P(X=x)
        P_x = (df[X].value_counts().div(len(df)))[x]

        for y in df[Y].unique():
            # Si la valeur de Y n'est pas associée à la valeur de X, on passe à la suivante
            if y not in dict_key_unique_values[x]:
                continue

            # Calcul de la probabilité conjointe P(X=x, Y=y)
            P_y = (df[Y].value_counts().div(len(df)))[y]
            Px_y = proba_table[x][y] * P_x  # ou P_y
            
            # Mise à jour de l'information mutuelle selon la formule de l'entropie
            information += Px_y * math.log(Px_y / (P_x * P_y), 2)

    # Retourne l'information mutuelle calculée
    return information





def ConditionalMutualInformation(df, x, y, z):
    '''
    Calcule l'information mutuelle conditionnelle entre les variables X et Y, sachant Z, dans un dataframe.

    @param1 df: DataFrame contenant les données.
    @param2 x: Nom de la première variable.
    @param3 y: Nom de la deuxième variable.
    @param4 z: Nom de la variable conditionnelle.

    @return Valeur de l'information mutuelle conditionnelle entre X et Y sachant Z.
    '''
    # Nombre total d'observations dans le dataframe
    n = df.shape[0]
    
    # Initialisation de l'information mutuelle conditionnelle
    information = 0

    # Parcours des valeurs uniques de X
    for i in df[x].unique():
        # Parcours des valeurs uniques de Y
        for j in df[y].unique():
            # Parcours des valeurs uniques de Z
            for k in df[z].unique():
                # Probabilité conditionnelle P(X=i | Z=k)
                pxz = len(df[(df[x] == i) & (df[z] == k)]) / n
                # Probabilité conditionnelle P(Y=j | Z=k)
                pyz = len(df[(df[y] == j) & (df[z] == k)]) / n
                # Probabilité marginale P(Z=k)
                pz = len(df[df[z] == k]) / n
                # Probabilité conjointe P(X=i, Y=j, Z=k)
                pxyz = len(df[(df[x] == i) & (df[y] == j) & (df[z] == k)]) / n
    
                # Mise à jour de l'information mutuelle conditionnelle
                if pxz != 0 and pyz != 0 and pz != 0 and pxyz != 0:
                    information += pxyz * math.log2(pz * pxyz / (pxz * pyz))

    # Retourne l'information mutuelle conditionnelle calculée
    return information


def MeanForSymetricWeights(a):
    '''
    Calcule la moyenne des poids dans une matrice symétrique en ignorant les éléments de la diagonale.

    @param1 a : Matrice symétrique.

    @return Moyenne des poids en dehors de la diagonale.
    '''
    mean = 0

    # Parcours des lignes
    for i in range(0, len(a)):
        # Parcours des colonnes
        for j in range(0, len(a[0])):
            # Accumulation des poids, en ignorant les éléments de la diagonale
            if i != j:
                mean += a[i][j]

    # Calcul de la moyenne en divisant par le nombre d'éléments - la diagonale
    mean /= (len(a) * len(a[0]) - len(a))

    # Retourne la moyenne calculée
    return mean



def SimplifyConditionalMutualInformationMatrix(a):
    '''
    Simplifie une matrice de l'information mutuelle conditionnelle en mettant à zéro les poids
    inférieurs à la moyenne des poids (hors diagonale).

    @param1 a : Matrice de l'information mutuelle conditionnelle.

    @return La matrice modifiée.
    '''
    # Calcul de la moyenne des poids dans la matrice (hors diagonale)
    mean = MeanForSymetricWeights(a)
    
    # Parcours des lignes
    for i in range(len(a)):
        # Parcours des colonnes
        for j in range(len(a[0])):
            # Mise à zéro des poids inférieurs à la moyenne (hors diagonale)
            if a[i][j] < mean and i != j:
                a[i][j] = 0

    # Retourne la matrice modifiée
    return a



def Kruskal(df, a):
    '''
    Applique l'algorithme de Kruskal pour trouver l'arbre de recouvrement de poids maximal
    dans le graphe représenté par la matrice de l'information mutuelle conditionnelle.

    @param1 df : DataFrame contenant les données.
    @param2 a : Matrice de l'information mutuelle conditionnelle.

    @return La liste des arcs de l'arbre de recouvrement de poids maximal.
    '''
    # Simplifie la matrice de l'information mutuelle conditionnelle
    SimplifyConditionalMutualInformationMatrix(a)
    
    # Initialisation du graph et de la liste des arcs
    graph = []
    listKruskal = []
    
    # Initialisation des groupes
    groupes = {i: i for i in range(len(df.columns))}
    
    # Liste des attributs
    liste = df.columns
    
    # Parcours de la matrice
    for i in range(0, len(a)):
        for j in range(0, len(a)):
            if a[j][i] != 0:
                graph.append([j, i, a[i][j]])
    
    # Tri du graph en fonction du poids, puis des attributs (ordre décroissant)
    graph = sorted(graph, key=lambda x: (x[2], x[0], x[1]), reverse=True)
    
    # Parcours des arcs du graph
    for x, y, poids in graph:
        # Si les sommets x et y ne sont pas dans le même groupe (pas de circuit formé)
        if groupes[x] != groupes[y]:
            # Ajout de l'arc à la liste des arcs de l'arbre de recouvrement
            listKruskal.append((liste[x], liste[y], poids))
            
            # Mise à jour des groupes
            ancien_groupe, nouveau_groupe = groupes[x], groupes[y]
            for sommet, groupe in groupes.items():
                if groupe == ancien_groupe:
                    groupes[sommet] = nouveau_groupe

    # Retourne la liste des arcs de l'arbre de recouvrement de poids maximal
    return listKruskal





def ConnexSets(list_arcs):
    """
    Crée une liste d'ensembles d'attributs connectés.

    @param1 list_arcs : Liste d'arcs dirigés entre les attributs, avec poids.

    @return Liste d'ensembles représentant des ensembles d'attributs connectés.
    """
    # Initialisation des ensembles connectés
    ensembles_connectes = []

    # Parcours des arcs et connexion des attributs
    for arc in list_arcs:
        source, destination, poids = arc

        source_present = None
        dest_present = None

        for ensemble_attributs in ensembles_connectes:
            if source in ensemble_attributs or destination in ensemble_attributs:
                source_present = ensemble_attributs

        if source_present is None:
            # Créer un nouvel ensemble si la source n'est dans aucun ensemble
            ensembles_connectes.append({source, destination})
        elif 'class' not in source_present:
            # Ajouter la destination à l'ensemble de source si la classe n'est pas impliquée
            source_present.add(destination)

    # Filtrer les ensembles pour ne pas inclure ceux qui contiennent uniquement la classe
    ensembles_connectes = [ensemble for ensemble in ensembles_connectes if 'class' not in ensemble]

    return ensembles_connectes


def OrientConnexSets(df, arcs, classe):
    '''
    Oriente les arcs pour chaque ensemble d'attributs connexes en utilisant l'information mutuelle
    entre chaque attribut et la classe.

    @param1 df : DataFrame contenant les données.
    @param2 arcs : Liste d'ensembles d'attributs connexes.
    @param3 classe : Nom de la variable classe.

    @return Liste des arcs orientés.
    '''
    oriented_arcs = []

    for connex_set in arcs:
        # Calcul de l'information mutuelle entre chaque attribut de l'ensemble et la classe
        mutual_infos = [(attr, MutualInformation(df, attr, classe)) for attr in connex_set]

        # Tri des attributs en fonction de l'information mutuelle (ordre décroissant)
        mutual_infos = sorted(mutual_infos, key=lambda x: x[1], reverse=True)

        # Choix de la racine comme l'attribut avec la plus haute information mutuelle
        root = mutual_infos[0][0]

        # Orientations des arcs de l'ensemble vers la racine
        for attr in connex_set:
            if attr != root:
                oriented_arcs.append((attr, root))

    return oriented_arcs





#####
#8- Conclusion finale
#####
# Le classifieur bayésiens présente une proximité remarquable avec la droite où Rappel égale Précision 
# Précision en abscisse et Rappel en ordonnée), indiquant des valeurs élevées tant pour la précision que pour le rappel. 
# Il se positionne graphiquement plus près du point idéal en haut à droite, 
# démontrant ainsi une efficacité supérieure par rapport aux autres classifieurs, comme explicité dans notre réponse à la question 7.1.
# D'une manière générale, il est observé que les classifieurs bayésiens fonctionnent de manière optimale lorsque les attributs utilisés présentent peu ou pas 
# d'influence mutuelle (dépendance), tout en exerçant une influence significative sur la variable cible (target).Une problématique rencontrée concerne les biais présents dans la base de données.
# Lorsqu'une valeur d'attribut apparaît dans le jeu de test sans avoir été préalablement observée dans le jeu d'entraînement, il devient impossible de déduire la valeur de la variable target pour l'individu en question. 
# Dans notre approche, nous avons supposé que si une valeur n'a jamais été rencontrée lors de l'entraînement, sa probabilité d'occurrence est nulle. D'autres hypothèses, potentiellement plus appropriées, pourraient être envisagées pour traiter de tels cas.
# Il est évident que la qualité des données utilisées est cruciale pour obtenir un classifieur à la fois efficace et le moins biaisé possible.
# Ces expériences ont également mis en évidence la puissance des outils et propriétés issus des statistiques/probabilités. Sans ces derniers, la plupart des calculs que nous avons effectués serait irréalisable sur une machine avec une approche naïve, étant donné la taille en mémoire des tables de probabilités.
#####


