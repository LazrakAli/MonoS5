from utils import AbstractClassifier, drawGraph
import pandas as pd
import math
import scipy
from statistics import mean
import matplotlib.pyplot as plt

def getPrior(data):
    '''
    Calcule la probabilité à priori ainsi qu'un intevalle de confiance à 95% autour de l'estimation
    @param df: le dataframe à étudier
    @return : un dictionnaire contenant 3 clés 'estimation', 'min5pourcent', 'max5pourcent'
    '''
    target_values = data['target'].tolist()
    avg = mean(target_values)
    variance = avg * (1 - avg)
    standard_deviation = math.sqrt(variance)
    #pour un intervalle de confiance à 95% on trouve z=1.96
    min5percent = avg - 1.96 * standard_deviation / math.sqrt(len(target_values))
    max5percent = avg + 1.96 * standard_deviation / math.sqrt(len(target_values))

    return {'estimation' : avg, 'min5pourcent' : min5percent, 'max5percent' : max5percent}

class APrioriClassifier(AbstractClassifier):
    '''
    Estime la classe d'un individu selon la probabilité à priori
    '''
    def __init__(self, attrs):
        '''
        La constructeur de APrioriClassifier
        @param attrs: le dictionnaire nom-valeur des attributs
        '''
        prior = getPrior(attrs)
        if (prior['estimation'] > 0.5):
            self.class_found = 1
        else:
            self.class_found = 0
			
	#Redéfinition des fonctions de la classe mère
    def estimClass(self, attrs):
        """
		à partir d'un dictionnaire d'attributs, estime la classe 0 ou 1

		@param attrs: le dictionnaire nom-valeur des attributs
		@return: la classe 0 ou 1 estimée
		"""

        return self.class_found

    def statsOnDF(self, df):
        '''
        Calcule la précision et le rappel du classifieur
        
        @return: le dictionnaire des statistiques de dictionnaire df
        '''
        vp=0
        vn=0
        fp=0
        fn=0
        precision=0
        rappel=0
        for t in df.itertuples():
            dic=t._asdict()
            dic.pop('Index',None)
            if dic['target']==1:
                if self.estimClass(dic)==1:
                    vp=vp+1
                else:
                    fn=fn+1
            else:
                if self.estimClass(dic)==1:
                    fp=fp+1
                else:
                    vn=vn+1
        precision=vp/(vp+fp)
        rappel=vp/(fn+vp)
        return {'VP':vp,'VN':vn,'FP':fp,'FN':fn,'Précision':precision,'Rappel':rappel}

def P2D_l(df, attr):
    '''
    Calcule dans la dataframe la probabilité P(attr|target) sous la forme
    d'un dictionnaire associant à la valeur 't' un dictionnaire associant à la
    valeur 'a' la probabilité P(attr = a | target = t)
    '''
    dict_ret = dict()
    possible_target_values = df.target.unique()
    for target_val in possible_target_values:
        df_target = df[df.target == target_val]
        nb_same_target = df_target.shape[0]
        dict_ret[target_val] = dict()

        possible_attr_values = df[attr].unique()
        for attr_val in possible_attr_values:
            df_same_attr = df_target[df_target[attr]==attr_val]
            nb_same_attr_val = df_same_attr.shape[0]
            proba = nb_same_attr_val / nb_same_target
            dict_ret[target_val][attr_val] = proba

    return dict_ret


def P2D_p(df, attr):
    '''
    Calcule dans le dataframe la probabilite P(target|attr) sous la forme
    d un dictionnaire associant à la valeur t un dictionnaire associant à la
    valeur 't' la probabilite P(target = t| attr = a)
    '''
    dict_ret = dict()
    possible_attr_values = df[attr].unique()
    for attr_val in possible_attr_values:
        df_same_attr = df[df[attr] == attr_val]
        nb_same_attr_val = df_same_attr.shape[0]
        dict_ret[attr_val] = dict()

        possible_target_values = df.target.unique()
        for target_val in possible_target_values:
            df_target = df_same_attr[df_same_attr.target==target_val]
            nb_same_target = df_target.shape[0]
            proba = nb_same_target/ nb_same_attr_val
            dict_ret[attr_val][target_val] = proba

    return dict_ret
            
class ML2DClassifier (APrioriClassifier):
    '''
    Classifie selon le maximum de vraisemblance
    '''
    def __init__(self, attrs, attr):
        self.probas = P2D_l(attrs, attr)
        self.attr = attr

    def estimClass(self, attrs):
        """
		à partir d'un dictionnaire d'attributs, estime la classe 0 ou 1

		:param attrs: le  dictionnaire nom-valeur des attributs
		:return: la classe 0 ou 1 estimée
		"""
        if self.probas[0][attrs[self.attr]] > self.probas[1][attrs[self.attr]]:
            return 0

        return 1
    
class MAP2DClassifier (APrioriClassifier):
    '''
    Classifie selon le maximum a posteriori (MAP)
    '''
    def __init__(self, attrs, attr):
        self.probas = P2D_p(attrs, attr)
        self.attr = attr

    def estimClass(self, attrs):
        """
		à partir d'un dictionnaire d'attributs, estime la classe 0 ou 1
		:param attrs: le  dictionnaire nom-valeur des attributs
		:return: la classe 0 ou 1 estimée
		"""
        if self.probas[attrs[self.attr]][0] > self.probas[attrs[self.attr]][1]:
            return 0

        return 1
    
    
def nbParams(df, attrs = None):
    '''
    Calcule la taille mémoire des tables P(target|attr1,..,attrk) étant donné un dataframe 
    et la liste [target,attr1,...,attrl] en supposant qu'un float est représenté sur 8 octets.
    Le résultat est directement affiché et non retourné.
    :param df: le dataframe contenant les données de la base
    :param attrs: la liste des attributs à considérer dans le calcul
    '''
    
    #Si on n'indique pas le nom des attributs, on calcule pour tous les atributs
    if attrs == None:
        attrs=[str(key) for key in df.keys()]
    
    cpt = 1
    for attr in attrs:
        #dic est un dictionnaire qui contient les attributs souhaités
        dic = {a:None for a in df[attr]}
        cpt= cpt*len(dic)

    #Affichage:
    nb_octets=cpt*8
    aff=""
    aff=aff+str(len(attrs))+" variable(s) : "+str(nb_octets)+" octets"

    s_o = ""
    s_ko=""
    s_mo=""
    s_go=""
  
    if nb_octets>=1024:
        aff = aff + " = "
        nb_ko=nb_octets//1024
        nb_octets=nb_octets%1024
        s_o = str(nb_octets) + "o"
        s_ko = str(nb_ko) + "ko "
        
        if nb_ko>=1024:
            nb_mo=nb_ko//1024
            nb_ko=nb_ko%1024
            s_ko = str(nb_ko) + "ko "
            s_mo = str(nb_mo) + "mo "
            
            if nb_mo>=1024:
                nb_go=nb_mo//1024
                nb_mo=nb_mo%1024	
                s_mo = str(nb_mo) + "mo "
                s_go = str(nb_go) + "go "

    aff=aff + s_go + s_mo + s_ko + s_o
    print(aff)
    
def nbParamsIndep(df, attrs = None):
    '''
    Calcule la taille mémoire nécessaire pour représenter les tables de probabilité 
    étant donné un dataframe, en supposant qu'un float est représenté sur 8octets et en supposant 
    l'indépendance des variables.
    Le résultat est directement affiché et non retourné.
    :param df: le dataframe contenant les données de la base
    :param attrs: la liste des attributs à considérer dans le calcul
    '''
    #Si on n'indique pas le nom des attributs, on calcule pour tous les atributs
    if attrs == None:
        attrs=[str(key) for key in df.keys()]

    cpt = 0
    for attr in attrs:
        #dic est un dictionnaire qui contient des attributs souhaités
        dic = {a: None for a in df[attr]}
        cpt=cpt+len(dic)

    #Affichage:
    nb_octets=cpt*8
    aff=""
    aff=aff+str(len(attrs))+" variable(s) : "+str(nb_octets)+" octets"

    s_o = ""
    s_ko=""
    s_mo=""
    s_go=""
    if nb_octets>=1024:
        aff = aff + " = "
        nb_ko=nb_octets//1024
        nb_octets=nb_octets%1024
        s_o = str(nb_octets) + "o"
        s_ko = str(nb_ko) + "ko "
    
        if nb_ko>=1024:
            nb_mo=nb_ko//1024
            nb_ko=nb_ko%1024
            s_ko = str(nb_ko) + "ko "
            s_mo = str(nb_mo) + "mo "
            if nb_mo>=1024:
                nb_go=nb_mo//1024
                nb_mo=nb_mo%1024
                s_mo = str(nb_mo) + "mo "
                s_go = str(nb_go) + "go "

    aff=aff + s_go + s_mo + s_ko + s_o
    print(aff)

def drawNaiveBayes(df, class_attr):
    '''
    Dessine un graphe représentant un modèle naïve bayes à partir d'un dataframe et du nom de l'attribut qui est la classe
    :param df: le dataframe contenant les données de la base
    :param class_attr: le nom de l'attribut-classe
    '''
    modelStr = ""
    for attr in df.columns :
        if (attr != class_attr) :
            modelStr += class_attr + "->" + attr + ";"
    return drawGraph(modelStr)

def nbParamsNaiveBayes(df, class_attr, attrs=None):
    '''
    Calcule la taille mémoire pour représenter les tables de probabilité en utilisant l'hypothèse du Naive Bayes
    Le résultat est directement affiché et non retourné.
    :param df: le dataframe contenant les données de la base
    :param class_attr: le nom de l'attribut-classe
    :param attrs: la liste des attributs à considérer dans le calcul
    '''
    if attrs == None : #si la liste d'attributs n'est pas précisée, on prend tous les attributs du df
        attrs = df.columns
    nb_possible_values_class_attr = len(df[class_attr].unique()) #nb valeurs possibles de l'attribut classe
    nb_values = 0 #nombre de valeurs des attributs
    for attr in attrs :
        if (attr != class_attr) :
            nb_possible_values = len(df[attr].unique())
            nb_values += nb_possible_values * nb_possible_values_class_attr
            
    nb_octets=(nb_possible_values_class_attr + nb_values ) * 8#(nb_valeurs_classe + nb_valeurs_attrs) * taille float
    aff=""
    aff+=str(len(attrs))+" variable(s) : "+str(nb_octets)+" octets"

    s_o = ""
    s_ko=""
    s_mo=""
    s_go=""
    if nb_octets>=1024:
        aff = aff + " = "
        nb_ko=nb_octets//1024
        nb_octets=nb_octets%1024
        s_o = str(nb_octets) + "o"
        s_ko = str(nb_ko) + "ko "
        if nb_ko>=1024:
            nb_mo=nb_ko//1024
            nb_ko=nb_ko%1024
            s_ko = str(nb_ko) + "ko "
            s_mo = str(nb_mo) + "mo "
            if nb_mo>=1024:
                nb_go=nb_mo//1024
                nb_mo=nb_mo%1024	
                s_mo = str(nb_mo) + "mo "
                s_go = str(nb_go) + "go "

    aff=aff + s_go + s_mo + s_ko + s_o
    print(aff)

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
        self.probas = dict() #clé=attr, valeur=P(attr|target)
        for attr in attrs :
            if attr != "target" :
                self.probas[attr] = P2D_l(attrs, attr)

        self.p_target_1 = getPrior(attrs)['estimation'] #p(target = 1)
        self.p_target_0 = 1 - self.p_target_1
        
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
                    proba_class_is_0 *= 0
                try:
                    proba_class_is_1 *= self.probas[attr][1][attrs[attr]]
                except KeyError :
                    proba_class_is_1 *= 0

        #p(attr1,attr2,...,attrn) = P(attr1|target=0)...P(attrn|target=0)P(target=0) + P(attr1|target=1)...P(attrn|target=1)P(target=1)
        if (proba_class_is_0 == 0 and proba_class_is_1 == 0): #si les deux probas sont égales on choisit toujours la classe 0
            proba_class_is_0 = 1
        p_attrs = (proba_class_is_0 * self.p_target_0 + proba_class_is_1 * self.p_target_1)

        return {0 : proba_class_is_0 * self.p_target_0 / p_attrs, 1 : proba_class_is_1 * self.p_target_1 / p_attrs} #formule Naive Bayes a posteriori

    def estimClass(self, attrs):
        """
		à partir d'un dictionnaire d'attributs, estime la classe 0 ou 1
		:param attrs: le  dictionnaire nom-valeur des attributs
		:return: la classe 0 ou 1 estimée
		"""
        proba_class = self.estimProbas(attrs)
        return max(proba_class, key=proba_class.get)

def isIndepFromTarget(df, attr, x):
    """
    à partir d'un dictionnaire d'attributs, pour l'attribut attr,
    mesure l'indépendance
    :param df: le  dictionnaire nom-valeur des attributs
           attr: le nom de l'attribut cherché
           x:La valeur pour évaluer l'indépendance
    :return: true si attr est indépendant de target, false sinon
    """
    contingency_tables = pd.crosstab(df['target'], df[attr])
    chi2, p, dof, ex =scipy.stats.chi2_contingency(contingency_tables)
    return p>=x


class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
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
                
                p0 = self.P2DL_dic[attr][0][attrs[attr]]
                p1 = self.P2DL_dic[attr][1][attrs[attr]]

                attr_0 = attr_0*p0
                attr_1 = attr_1*p1

        self.dic[0] = attr_0
        self.dic[1]= attr_1
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

def mapClassifiers(dic,df):
    '''
    À partir d'un dictionnaire dic de classifiers et d'un dataframe df, représente graphiquement
    les classifiers dans l'espace (précision,rappel)
    :param dic: dictionnaire de classifieurs
    :param df: le dataframe contenant les données de la base
    '''
    plt.figure(figsize=(7,5))
  
    for key in dic:
        cl=dic[key].statsOnDF(df)
        precision=cl['Précision']
        rappel=cl['Rappel']
        print(key,": ",precision,rappel)
        plt.scatter(precision,rappel,marker='x',color='r')
        plt.annotate(key,(precision,rappel))
  
def MutualInformation(df,X,Y):
    """
    Calcule l'information mutuelle conditionnelle entre les variables X et Y, sachant Z, dans un dataframe.

    @param1 df: DataFrame contenant les données.
    @param2 X: Nom de la première variable.
    @param3 Y: Nom de la deuxième variable.
    @param4 Z: Nom de la variable conditionnelle.

    @return: Valeur de l'information mutuelle conditionnelle entre X et Y sachant Z.
    """
    proba_table = df.groupby(X)[Y].value_counts() / df.groupby(X)[Y].count()

    information = 0.0

    list_values_index = proba_table.index.values.tolist()
    dict_key_unique_values = {}

    for x in df[X].unique():
        dict_key_unique_values[x] = []

    for (x,y) in list_values_index:
        dict_key_unique_values[x].append(y)

    for x in df[X].unique():

        P_x = (df[X].value_counts().div(len(df)))[x]

        for y in df[Y].unique():

            if y not in dict_key_unique_values[x]:
                continue

            P_y = (df[Y].value_counts().div(len(df)))[y]
            Px_y = proba_table[x][y] * P_x # ou P_y
            
            information += Px_y * math.log(Px_y/(P_x * P_y) ,2)

    return information

def ConditionalMutualInformation(df,X,Y,Z):

    mutualInformation = 0.0

    proba_table_Z_Y = df.groupby(Z)[Y].value_counts() / df.groupby(Z)[Y].count()
    proba_table_Y_X = df.groupby(Y)[X].value_counts() / df.groupby(Y)[X].count()

    dict_key_unique_values_Z_Y = {}
    dict_key_unique_values_Y_X = {}

    list_values_index_Z_Y = proba_table_Z_Y.index.values.tolist()
    list_values_index_Y_X = proba_table_Y_X.index.values.tolist()
    

    # Z sera la target
    # on sait que X ou Y est independant de Z = la target , on peut donc ecrire
    # P(X,Y,Z) = P(X) * P(Y|X) * P(Z|Y) ou P(X,Y,Z) = P(Y) * P(X|Y) * P(Z|X)

    for z in df[Z].unique(): 
        dict_key_unique_values_Z_Y[z] = []

    for y in df[Y].unique():
        dict_key_unique_values_Y_X[y] = []

    for (z,y) in list_values_index_Z_Y:
        dict_key_unique_values_Z_Y[z].append(y)

    for (y,x) in list_values_index_Y_X:
        dict_key_unique_values_Y_X[y].append(x)

    for z in df[Z].unique():

        P_z = (df[Z].value_counts().div(len(df)))[z]


        for x in df[X].unique():

            if x not in dict_key_unique_values_Y_X[x]:
                continue

            P_x = (df[X].value_counts().div(len(df)))[x]
            
            for y in df[Y].unique():

                if y not in dict_key_unique_values_Z_Y[y]:
                    continue

                P_y = (df[X].value_counts().div(len(df)))[x]

                # P_y_x = P(Y|X)
                Py_x =  proba_table_Y_X[x][y]
                Pz_y = proba_table_Z_Y[z][y]

                Px_y_z = P_x * Py_x * Pz_y

    return mutualInformation

