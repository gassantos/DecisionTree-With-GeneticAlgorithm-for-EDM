#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import time
import itertools as itools
import graphviz as gvz
import numpy as np
import pandas as pd
import random as rdm
import matplotlib.pylab as plt
import seaborn as sns

from sklearn import tree as T
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split as split, StratifiedKFold as SKFold, GridSearchCV as gridSCV
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report 
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, cohen_kappa_score, precision_score
from sklearn.feature_selection import RFECV

#from pandas_ml import ConfusionMatrix
from deap import creator, base, tools, algorithms
from scipy import interpolate, stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

import itertools, graphviz, mpld3, requests, json, time, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)



import platform, os
if platform.system() == 'Windows' :
    APP='C:/Program Files (x86)/Graphviz2.38/bin/'
    os.environ["PATH"] += os.pathsep + APP


# # Serving by API Flask to Ruby on Rails Code

# In[2]:


from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

hello()


# # CEP das Localidades de Ensino da UFF
# 
# * CEP - VALONGUINHO:             24020140
# * CEP - PRAIA VERMELHA:          24210200
# * CEP - BIOMEDIO:                24210130
# * CEP - MEDICINA:                24020071
# * CEP - IACS:                    24210510
# * CEP - REITORIA:                24220900
# * CEP - ENGQUIMICA:              24210346
# * CEP - VOLTA-REDONDA:           27213145
# * CEP - MACAÉ:                   27930560
# * CEP - RIO DAS OSTRAS:          28890000
# * CEP - PETROPOLIS:              25650050
# * CEP - SANTO ANTONIO DE PADUA:  28470000
# * CEP - Campos dos Goytacazes:
# * CEP - Nova Iguaçu:
# * CEP - Angra Dos Reis:
# * CEP - Itaperuna:
# * CEP - Miracema:
# * CEP - Pinheiral:
# * CEP - Bom Jesus do Itabapoana:
# * CEP - Cabo Frio:
# * CEP - São João De Meriti:
# * CEP - Arraial Do Cabo:
# * CEP - Quissamã:
# * CEP - Nova Friburgo:
# * CEP - Petrópolis:
# * CEP - Oriximiná:

# # Functions

# ### a) Distance Calculation to College

# In[3]:


def calculaDistancia(cepOrigem, cepDestino):
    
    KEY ='key=AIzaSyAJqU9516xSUEDnojnedeB3AbfXCPnxrCw'
    GOOGLEAPI = 'https://maps.googleapis.com/maps/api/distancematrix/json?'+KEY+'&origins='
    
    getOrigem  = 'https://viacep.com.br/ws/'+cepOrigem+'/json/'
    getDestino = 'https://viacep.com.br/ws/'+cepDestino+'/json/'
    
    origem  = json.loads(requests.get(getOrigem).text)
    destino = json.loads(requests.get(getDestino).text)
    
    strOrigem = origem['localidade']+' '+origem['uf'].replace(' ', '+')  
    strDestino = destino['localidade']+' '+destino['uf'].replace(' ', '+')
    
    return requests.get(GOOGLEAPI+strOrigem+'&destinations='+strDestino+'&mode=bicycling')                                                                        


# ### a) Plotting Confusion Matrix

# In[4]:


def plotConfusionMatrix(matrix, target_names, title="\nMatriz de Confusão\n", cmap=None, accuracy=None, normalize=True):
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()

    if target_names is not None:
        marks = np.arange(len(target_names))
        plt.xticks(marks, target_names) #, rotation=30)
        plt.yticks(marks, target_names)

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]


    thresh = matrix.max() / 1.5 if normalize else matrix.max()
    for i, j in itools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2}".format(matrix[i, j]),
                     horizontalalignment="center",
                     color="red" if matrix[i, j] <= thresh else "red")
        else:
            plt.text(j, i, "{:,}".format(matrix[i, j]),
                     horizontalalignment="center",
                     color="red" if matrix[i, j] > thresh else "red")

    if accuracy is None:
        accuracy = np.trace(matrix) / float(np.sum(matrix))
    
    misclass = 1 - accuracy
    
    plt.figure(1)
    plt.tight_layout()
    plt.xlabel('\n\naccuracy={:0.6f}; misclass={:0.6f}'.format(accuracy, misclass))
    plt.show()


# ### b) Feature subset fitness function with Machine Learning Techniques

# In[5]:


def getFitnessMLT(individual, X_train, X_test, y_train, y_test):

    # Extracting feature columns that we do not use and apply category mapping to the features
    notFeatures = [idx for idx in range(len(individual)) if individual[idx] == 0]
    X_treinoNotFeatures = X_train.drop(X_train.columns[notFeatures], axis=1)
    X_treinoFeatures = pd.get_dummies(X_treinoNotFeatures)
    X_testeNotFeatures = X_test.drop(X_test.columns[notFeatures], axis=1)
    X_testeFeatures = pd.get_dummies(X_testeNotFeatures)

    # Remove any columns that not are in both the training and test sets
    comumFeatures = set(X_treinoFeatures.columns) & set(X_testeFeatures.columns)
    treinoFeatures = set(X_treinoFeatures.columns) - comumFeatures
    X_treinoFeatures = X_treinoFeatures.drop(list(treinoFeatures), axis=1)
    testeFeatures = set(X_testeFeatures.columns) - comumFeatures
    X_testeFeatures = X_testeFeatures.drop(list(testeFeatures), axis=1)

    # Apply Decision Tree on the data, and calculate accuracy
    clf = T.DecisionTreeClassifier()
    clf.fit(X_treinoFeatures, y_train)
    predictions = clf.predict(X_testeFeatures)
    accuracy = accuracy_score(y_test, predictions)
    #precision = precision_score(y_test, predictions)
    #fscore = f1_score(y_test, predictions) # Teve um desempenho pior, se comparado à acurácia

    # Return calculated accuracy as fitness
    return (accuracy, )


# ### c) Obtaining the Best Individuals  to Genetic Algorithm

# In[6]:


def getHof():

    # Initialize population
    numPop = 100
    numGen = 10
    pop = toolbox.population(n=numPop)
    hof = tools.HallOfFame(numPop * numGen)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Launch genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=numGen, stats=stats, halloffame=hof, verbose=True)

    # Return the hall of fame
    return hof


# ### c) Get list of percentiles in the hall of fame

# In[7]:


def percentHof(hof):
    percentileList = [i / (len(hof) - 1) for i in range(len(hof))]
    return percentileList


# ### d) Get Accuracies List of the HOF (hall of fame)

# In[8]:


def getMetrics(hof):
    
    # Gather fitness data from each percentile
    testAccuracyList = []
    validationAccuracyList = []
    individualList = []
    
    for individual in hof:
        testAccuracy = individual.fitness.values
        validationAccuracy = getFitnessMLT(individual, X_trainAndTest, X_validation, y_trainAndTest, y_validation)
        testAccuracyList.append(testAccuracy[0])
        validationAccuracyList.append(validationAccuracy[0])
        individualList.append(individual)
    
    testAccuracyList.reverse()
    validationAccuracyList.reverse()
    individualList.reverse()
    
    return testAccuracyList, validationAccuracyList, individualList


# ### e) Function List Best Performance 

# In[9]:


def listBestPerformance(validAccList):
    
    maxValAccIndices = [] 
    maxValIndividuos = []
    maxBetterSubset = []
    maxWorseSubset = []
    
    for idx in range(len(validAccList)): 
        if validAccList[idx] == max(validAccList):
            maxValAccIndices = [idx]
            
    for idx in maxValAccIndices:
        maxValIndividuos = [individualList[idx]]
        
    for individual in maxValIndividuos:
        for index in range(len(individual)): 
            if individual[index] == 1:
                maxBetterSubset += [list(allFeatures)[index]]
            else: 
                maxWorseSubset += [list(allFeatures)[index]]
    
    return maxValAccIndices, maxValIndividuos, maxBetterSubset, maxWorseSubset


# ### f) Function Optimal Feature Subset

# In[10]:


def getWorseFeatures(validAccList):
    
    # Obtaining list of best performance individuals
    maxValAccIndices, maxValIndividuos, maxBetterSubset, maxWorseSubset = listBestPerformance(validAccList)

    bestIndividualList = []
    maxValidAccFS = None
    
    for idx in range(len(maxValAccIndices)): 
        maxValidAccFS = validAccList[maxValAccIndices[idx]]
        bestIndividualList = maxValIndividuos[idx]
    
    print('\nValidation Accuracy with AF: \t\t' +str(validationAccuracyAll[0]))
    print('Validation Accuracy with FS: \t\t' +str(maxValidAccFS))
    print('Best Individual: \t\t\t' +str(bestIndividualList))
    print('Number of Selected Features: \t\t' +str(len(maxBetterSubset)))
    print('Number of Not Selected Features: \t' +str(len(maxWorseSubset))+'\n')
    print('Set of Selected Features:\n' +str(maxBetterSubset)+'\n')
    print('Set of UnSelected Features:\n' +str(maxWorseSubset)+'\n')
    
    return maxWorseSubset


# ### g) Function Ranking of Best Individuals

# In[11]:


def getBestRanking(testAccList, validAccList, individualList):
    
    # Gather fitness data from each percentile
    testAccuracyRanking = []
    validationAccuracyRanking = []
    bestIndividualRanking = []
    
    # Variables for Accuracy List
    it = 0
    LIMIT = len(validAccList)
    
    # Iterations for Ranking of the Best Individuals
    for individual in individualList:
        testAccuracy = individual.fitness.values
        
        if it < LIMIT :
            if validAccList[it] >= 0.99 :
                testAccuracyRanking.append(testAccuracy[0])
                validationAccuracyRanking.append(validAccList[it])
                bestIndividualRanking.append(individual)
        else:
            break
        
        it=it+1
    
    return bestIndividualRanking, validationAccuracyRanking, testAccuracyRanking


# ### h) Function Optimal Number Features through sklearn.RFECV

# In[12]:


def getOptimalNumberFeatures(X, y):
    
    
    for c in X.columns:
        if X[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(X[c].values))
            X[c] = lbl.transform(list(X[c].values))
    
    # The accuracy scoring is proportional to the number of correct classifications
    rfecv = RFECV(estimator=DecisionTreeClassifier(), step=1, cv=SKFold(5), scoring='accuracy')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    return  rfecv.n_features_


# # Importando o Dataset de Alunos

# In[13]:


starTime = time.time()
data = pd.read_csv('DATASET_ALUNOS_FORMADOS-EVADIDOS_2012-2018.csv', sep=';')
data.head(20)


# In[14]:


# 23 - Admininstração; 7 - Direito; 42 - Eng. Produção; 31 - Ciências da Computação;

flagCurso = False
if flagCurso :
    dataCurso = data.query('CURSO == 23 & ANOINGRESSO <= 2014') #
    print(dataCurso.shape)
    dataCurso.head()

else :
    dataCurso = data.copy()
    print(dataCurso.shape)
    dataCurso.head()


# # 1) Descriptive Statistics

# In[15]:


### Dropping some features
dataCurso = dataCurso.drop(columns=['MATRICULA', 'PERIODODISC', 'DISCIPLINA', 'NOTADISC', 'RESULTDISC'])
dataCurso.head()


# In[16]:


### Remove duplicates
dataCurso = dataCurso.drop_duplicates(keep='first')
print(dataCurso.shape)
dataCurso.head()


# ### 1) Ethnicity:

# In[17]:


dataCurso.groupby(['ANOINGRESSO']).agg(['count', 'median'])


# In[18]:


dataCurso.groupby(['COR']).agg(['count', 'median'])


# ### 2) Gender:

# In[19]:


dataCurso.groupby(['SEXO']).agg(['count', 'median'])


# # Formas de Concorrências dos Candidatos à Graduação
# 
# * AC - Candidatos de ampla concorrência
# * L1 - Candidatos com renda familiar bruta per capita igual ou inferior a 1,5 salário mínimo que tenham cursado integralmente o ensino médio em escolas públicas (Lei nº 12.711/12).
# * L2 - Candidatos autodeclarados pretos, pardos ou indígenas, com renda familiar bruta per capita igual ou inferior a 1,5 salário mínimo e que tenham cursado integralmente o ensino médio em escolas públicas (Lei nº 12.711/2012).
# * L3 - Candidatos que, independentemente da renda (art. 14, II, Portaria Normativa nº 18/2012), tenham cursado integralmente o ensino médio em escolas públicas (Lei nº 12711/2012).
# * L4 - Candidatos autodeclarados pretos, pardos ou indígenas que, independentemente da renda (art. 14, II, Portaria Normativa nº 18/2012), tenham cursado integralmente o ensino médio em escolas públicas (Lei nº 12.711/2012).
# * A1 - Candidatos que cursaram todo o ensino médio em estabelecimento da rede pública estadual ou municipal de qualquer unidade da federação, excluídos os colégios federais, universitários, militares e de aplicação ¿ Política de Ação Afirmativa da UFF.
# * A0 - Ampla concorrência
# * L5 - Candidatos que, independentemente da renda (art. 14, II, Portaria Normativa nº 18/2012), tenham cursado integralmente o ensino médio em escolas públicas (Lei nº 12.711/2012). 
# * L6 - Candidatos autodeclarados pretos, pardos ou indígenas que, independentemente da renda (art. 14, II, Portaria Normativa nº 18/2012, tenham cursado integralmente o ensino médio em escolas públicas (Lei nº 12.711/2012). 
# * L9 - Candidatos com deficiência que tenham renda familiar bruta per capita igual ou inferior a 1,5 salário mínimo e que tenham cursado integralmente o ensino médio em escolas públicas (Lei nº 12.711/2012).
# * L10 - Candidatos com deficiência autodeclarados pretos, pardos ou indígenas, que tenham renda familiar bruta per capita igual ou inferior a 1,5 salário mínimo e que tenham cursado integralmente o ensino médio em escolas públicas (Lei nº 12.711/2012).
# * L13 - Candidatos com deficiência que, independentemente da renda (art. 14, II, Portaria Normativa nº 18/2012), tenham cursado integralmente o ensino médio em escolas públicas (Lei nº 12.711/2012).
# * L14 - Candidatos com deficiência autodeclarados pretos, pardos ou indígenas que, independentemente da renda (art. 14, II, Portaria Normativa nº 18/2012), tenham cursado integralmente o ensino médio em escolas públicas (Lei nº 12.711/2012).
# 

# ### 3) Social Programs:

# In[20]:


dataCurso.groupby(['ACAOAFIRMATIVA']).agg(['count', 'median'])


# In[21]:


dataCurso.head(20)


# ### 4) Marital Status:

# In[22]:


dataCurso.groupby(['ESTADOCIVIL']).agg(['count', 'median'])


# In[23]:


# REMOVENDO AS COLUNAS BAIRRO, CEP E CIDADE
dataCurso = dataCurso.drop(columns=['BAIRRO', 'CEP', 'CIDADE'])
dataCurso.head()


# In[24]:


print(dataCurso.shape)
dataUnique = dataCurso.drop_duplicates(keep='first')
print(dataUnique.shape)


# In[25]:


print(dataUnique.COR.dtype)
np.unique(dataUnique.COR, return_counts=True)


# In[26]:


print(dataUnique.ACAOAFIRMATIVA.dtype)
np.unique(dataUnique.ACAOAFIRMATIVA, return_counts=True)


# In[27]:


total, card = dataUnique.shape
dataUnique.head()


# # 2) Descriptive Analisys

# In[28]:


dataUnique.describe()


# In[29]:


dataUnique.corr()


# In[30]:


dataUnique = dataUnique.drop(columns=['CURSO','MOBILIDADE'])
dataUnique.head()


# ## Majority Class

# #### Observation to Dropout and Conclusion

# In[31]:


dataUnique.groupby(['STATUSFORMACAO']).agg(['count', 'median'])


# In[32]:


dataEvad = len(dataUnique.query('STATUSFORMACAO == "EVADIDO"'))
print("Dropout Percent: ", (dataEvad/total)*100)

dataForm = len(dataUnique.query('STATUSFORMACAO == "FORMADO"'))
print("Graduated Percent: ", (dataForm/total)*100)

dataAtivo = len(dataUnique.query('STATUSFORMACAO == "ATIVO"'))
print("Enrolled Percent: ", (dataAtivo/total)*100)


# # Preprocessing to Data Visualization

# In[33]:


# Agora vamos remover os alunos ATIVOS do Dataset
dataUnique = dataUnique[dataUnique.STATUSFORMACAO != 'ATIVO']
print(dataUnique.shape)


# # Encoded to classification labels

# In[34]:


for c in dataUnique.columns:
    if dataUnique[c].dtype == 'object' and c == 'STATUSFORMACAO':
        lbl = LabelEncoder()
        lbl.fit(list(dataUnique[c].values))
        dataUnique[c] = lbl.transform(list(dataUnique[c].values))


# In[35]:


# Efetuando Mapeamento Categórico
dataUnique.head()


# ## Grade Point Average (GPA)

# #### Detailled statistics to GPA

# In[36]:


dataUnique.CR.describe()


# ## Ethnicity

# In[37]:


dataUnique.groupby(['COR']).agg(['count','mean', 'median'])


# In[38]:


dataUnique.groupby(['ACAOAFIRMATIVA']).agg(['count','mean', 'median'])


# ## Gender

# In[39]:


dataUnique.groupby(['SEXO']).agg(['count','mean', 'median'])


# ## Age

# In[40]:


dataUnique.groupby(['IDADE']).agg(['count','mean', 'median'])


# # Data Visualization

# #### GPA Distribuition

# In[41]:


plt.figure(figsize=(10,8))
sns.distplot(dataUnique.CR, hist=True,bins=10)
plt.show()


# #### Ethnicity Distribuition to Academic Performance

# In[42]:


plt.figure(figsize=(10,8))
sns.boxenplot(dataUnique.COR, dataUnique.CR)
plt.show()

plt.figure(figsize=(10,8))
sns.boxenplot(dataUnique.TEMPOPERMANENCIA, dataUnique.CR)
plt.show()

plt.figure(figsize=(10,8))
sns.boxenplot(dataUnique.CODTURNOATUAL, dataUnique.CR)
plt.show()


# In[43]:


#### Ethnicity Distribuition to Graduation Conclusion by Gender

plt.figure(figsize=(10,8))
sns.violinplot(dataUnique.COR, dataUnique.STATUSFORMACAO, hue=data.SEXO, split=True)
plt.show()

#### Social Programs Visualization to Academic Performance

plt.figure(figsize=(10,8))
sns.boxenplot(dataUnique.ACAOAFIRMATIVA, dataUnique.CR)
plt.show()

#### Social Programs Visualization to Graduation Conclusion

plt.figure(figsize=(10,8))
sns.violinplot(dataUnique.ACAOAFIRMATIVA, dataUnique.STATUSFORMACAO, hue=dataUnique.SEXO, split=True)
plt.show()


# In[44]:


#### Enrollment Year vs GPA Boxplot

plt.figure(figsize=(10,8))
sns.boxenplot(dataUnique.ANOINGRESSO, dataUnique.CR)
plt.show()

sns.jointplot(dataUnique.ANOINGRESSO, y=dataUnique.CR, data= dataUnique, kind='kde')
plt.show()

#### Ethnicity vs GPA Boxplot

plt.figure(figsize=(10,8))
sns.boxplot(dataUnique.COR, dataUnique.CR)
plt.show()


# In[45]:


#### GPA vs Gender vs Ethncity in Violinplot

# Draw a nested violinplot and split the violins for easier comparison
plt.figure(figsize=(10,8))
sns.violinplot(x="COR", y="CR", hue="SEXO", split=True, inner="quart", data=dataUnique)
sns.despine(left=True)
plt.show()

# Draw a nested violinplot and split the violins for easier comparison
plt.figure(figsize=(10,8))
sns.violinplot(x="COR", y="CR", hue="STATUSFORMACAO", split=True, inner="quart", data=dataUnique)
sns.despine(left=True)
plt.show()


# In[46]:


# Draw a nested violinplot and split the violins for easier comparison
plt.figure(figsize=(10,8))
sns.violinplot(x="ACAOAFIRMATIVA", y="CR", hue="STATUSFORMACAO", split=True, inner="quart", data=dataUnique)
sns.despine(left=True)
plt.show()

# Draw a nested violinplot and split the violins for easier comparison
plt.figure(figsize=(10,8))
sns.violinplot(x="SEXO", y="CR", hue="STATUSFORMACAO", split=True, inner="quart", data=dataUnique)
sns.despine(left=True)
plt.show()

# Draw a nested violinplot and split the violins for easier comparison
plt.figure(figsize=(10,8))
sns.violinplot(x="ESTADOCIVIL", y="CR", hue="STATUSFORMACAO", split=True, inner="quart", data=dataUnique)
sns.despine(left=True)
plt.show()


# # Correlation Heatmap

# In[47]:


# Compute the correlation matrix
corr = dataUnique.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# # Statistics Tests - ANOVA and Tukey
# 
# A **Análise de Variância (``ANOVA``)** testa se a média de alguma variável numérica difere-se nos níveis de significância de uma variável categórica. Essencialmente, responde à pergunta: algum dos meios do grupo difere um do outro? 
# 
# Não entraremos nos detalhes da realização de uma ANOVA à mão, pois ela envolve mais cálculos do que o *teste T*, 
# mas o processo é semelhante: você passa por vários cálculos para chegar a uma estatística de teste e depois 
# compara a estatística de teste para um valor crítico baseado em uma distribuição de probabilidade. 
# No caso da ANOVA, você usa a **``distribuição f``**.
# 
# A **biblioteca ``scipy``** tem uma função para realizar testes ANOVA unidirecionais chamados ``scipy.stats.f_oneway ()``. Vamos gerar uma certa idade de eleitor e dados demográficos e usar a ANOVA para comparar as idades médias entre os grupos:

# ### Comparação de médias: `Teste de Turking`

# In[48]:


print(pairwise_tukeyhsd(dataUnique.ANOINGRESSO, dataUnique.COR))


# In[49]:


print(pairwise_tukeyhsd(dataUnique.STATUSFORMACAO, dataUnique.SEXO))


# In[50]:


print(pairwise_tukeyhsd(dataUnique.STATUSFORMACAO, dataUnique.ANOINGRESSO))


# In[51]:


print(pairwise_tukeyhsd(dataUnique.STATUSFORMACAO, dataUnique.COR))


# ### Comparação de médias: ``MultiComparison``

# In[52]:


mc = MultiComparison(dataUnique.STATUSFORMACAO, dataUnique.COR)
plt.figure(figsize=(10,8))
mc.tukeyhsd().plot_simultaneous()
plt.show()


# # Get classes and one hot encoded feature vectors

# In[53]:


allClasses = dataUnique['STATUSFORMACAO'].values
allFeatures = dataUnique.drop(['STATUSFORMACAO'], axis=1)


# In[54]:


allFeatures.head()


# In[55]:


allClasses


# In[56]:


print("Quantidade de Campos NaN no Dataset: \n")
print(allFeatures.isnull().sum())


# In[57]:


print("Quantidade de Campos NaN no Dataset: \n")
allFeatures = allFeatures.fillna(1000)
print(allFeatures.isnull().sum())


# In[58]:


allFeatures.dtypes


# In[59]:


print(allFeatures.shape)
allFeatures.head()


# # Form training, test, and validation sets

# In[60]:


X_trainAndTest, X_validation, y_trainAndTest, y_validation = split(allFeatures, allClasses, test_size=0.20, random_state=42)
print(X_trainAndTest.shape)
print(X_validation.shape)
print(y_trainAndTest.shape)
print(y_validation.shape)


# In[61]:


X_train, X_test, y_train, y_test = split(X_trainAndTest, y_trainAndTest, test_size=0.20, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # First, we will apply logistic regression using all the features to acquire a baseline accuracy

# In[62]:


individual = [1 for i in range(len(allFeatures))]
testAccuracyAll = getFitnessMLT(individual, X_train, X_test, y_train, y_test)
validationAccuracyAll = getFitnessMLT(individual, X_trainAndTest, X_validation, y_trainAndTest, y_validation)


# In[63]:


print('Test accuracy with all features: \t' + str(testAccuracyAll[0]))
print('Validation accuracy with all features: \t' + str(validationAccuracyAll[0]))


# # Constructing a Model for Genetic Programing with DEAP Framework

# In[64]:


# Create Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# In[65]:


# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", rdm.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(dataUnique.columns) - 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# In[66]:


# Defining operators into an toolbox
toolbox.register("evaluate", getFitnessMLT, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


# # Now, we will apply a genetic algorithm to choose the best generations of individuals that had a better accuracy than the baseline

# In[67]:


start_time = time.time()
hof = getHof()
print("--- %s seconds ---" % (time.time() - start_time))


# In[68]:


# Obtaining all metrics of Hall of Fame
start_time = time.time()
testAccList, validAccList, individualList = getMetrics(hof)
print("--- %s seconds ---" % (time.time() - start_time))


# # Get ranking of the Best Individuals

# In[69]:


bestIndividualRanking, validAccRanking, testAccRanking = getBestRanking(testAccList, validAccList, individualList)


# # Get a list of best performance of Feature Selection

# In[70]:


start_time = time.time()
featureNotSelected = getWorseFeatures(validAccRanking)
print("--- %s seconds ---" % (time.time() - start_time))


# # Comparasion between Approaches: EduvolutionML vs Sklearn RCFV

# In[71]:


getOptimalNumberFeatures(allFeatures, allClasses)


# # Calculate best fit line for validation classification accuracy (non-linear)

# In[72]:


percentileList = percentHof(hof)
curva = interpolate.splrep(percentileList, validAccList, s=5.0)
evaluate = interpolate.splev(percentileList, curva)


# In[73]:


plt.figure(figsize=(8,6))
plt.figure(1)
plt.plot(percentileList, validAccList, marker='o', color='r')
plt.plot(percentileList, evaluate, color='b')
#plt.title('Validation Set Classification Accuracy vs. \n Continuum with Cubic-Spline Interpolation')
plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
plt.ylabel('Validation Set Accuracy')
plt.show()


# In[74]:


plt.figure(figsize=(8,6))
plt.figure(1)
plt.scatter(percentileList, testAccList)
plt.plot(percentileList, evaluate, color='r')
#plt.title('Validation Set Classification Accuracy vs. Continuum')
plt.xlabel('Population Ordered By Increasing Validation Set Accuracy')
plt.ylabel('Test Set Accuracy')
plt.show()


# # Building a Decision Tree Model with Feature Selection

# In[75]:


features = allFeatures.drop(columns=featureNotSelected)
print(features.shape)
features.head()


# In[76]:


nameFeatures = list(features)
nameFeatures


# In[77]:


for c in features.columns:
    if features[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(features[c].values))
        features[c] = lbl.transform(list(features[c].values))


# In[78]:


print(features.dtypes)


# In[79]:


features = np.array(features.values, dtype=np.float64)
print(features)
print(features.shape)


# # Data Preprocessing Unsupervised Stratified using Cross Validation

# In[80]:


names = ["KNN","LinearSVM","RBF-SVM","DecisionTree","RandomForest","NeuralNet","AdaBoost","NaiveBayes","QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]


# In[81]:


# Data Klustering
KCLUSTER=10
kmeans = KMeans(n_clusters=KCLUSTER, init='random')
kmeans.fit(features)

# Get groups
kmeans.cluster_centers_
groups = kmeans.labels_

# Training through Unsupervised Stratified KFold
bestAccuracyFolds = {}
bestAccuracyFolds['Model'] = 'Classifier','Accuracy','MCC','ROC','AUC-ROC','Report','Precision','F-Score'
maxAccuracyFold = {}

bestAccuracy = 0.0
bestClassifier = None
#bestMeanAccuracy = {}
bestMCC = 0.0
bestROC = 0.0
bestReport = None

for k in range(KCLUSTER):
    print('\nFOLD',k)
    X_train, X_validation, y_train, y_validation = split(features, allClasses, random_state=k, stratify=groups)

    c = 0
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_validation)
        accuracy = accuracy_score(y_validation, prediction)
        mcc = matthews_corrcoef(y_validation, prediction)
        roc = roc_curve(y_validation, prediction)
        rauc = roc_auc_score(y_validation, prediction)
        report = classification_report(y_validation, prediction)
        precision = precision_score(y_validation, prediction)
        fscore = f1_score(y_validation, prediction)
        kappa = cohen_kappa_score(y_validation, prediction)
        matrix = confusion_matrix(y_validation, prediction)
        
        if accuracy >= bestAccuracy:
            bestAccuracy = accuracy
            bestClassifier = clf
            #maxAccuracyFold[k] = X_train, y_train, X_validation, y_validation
            #bestMCC = mcc
            #bestROC = roc
            #bestReport = report
            
            bestAccuracyFolds[name] = clf, accuracy, mcc, roc, rauc, report, precision, fscore, kappa, matrix
        
        print(c,'\t-',name,'teve acurácia de', str(accuracy))
        c=c+1
    


# In[82]:


bestAccuracyFolds


# In[83]:


bestClassifier


# In[84]:


bestAccuracy


# In[85]:


clfDT = None 
accuracy = None
mcc = None
roc = None
report = None
precision = None
fscore = None
kappa = None
matrix = []

if bestAccuracyFolds.get('DecisionTree') != None :
    clfDT, accuracy, mcc, roc, rauc, report, precision, fscore, kappa, matrix = bestAccuracyFolds.get('DecisionTree')

else :
    clfDT = T.DecisionTreeClassifier()
    clfDT = clfDT.fit(X_train, y_train)
    print(clfDT)


# # Visualization of The Better Classifier

# In[86]:


dataClass = T.export_graphviz(clfDT, out_file=None, filled=True, rounded=True, proportion=True,
            special_characters=True, feature_names=nameFeatures, class_names=['Evadido','Formado'], node_ids=True)  
graph = gvz.Source(dataClass)
graph.render("clfDTfeatureSelection") # Save in PDF


# In[87]:


# Prediction for Decision Tree classifier with criterion as gini index
y_pred = clfDT.predict(X_validation)
print(y_pred)


# In[88]:


# Accuracy for Decision Tree classifier with criterion as gini index
if accuracy == None :
    accuracy = accuracy_score(y_validation, y_pred)*100
print("\nAccuracy is", accuracy)


# In[89]:


# Evaluate the Matthew’s correlation coefficient (MCC) for binary classes
if mcc == None :
    mcc = matthews_corrcoef(y_validation, y_pred)
print("\nMatthews Correlation Coeficient is",mcc)


# In[90]:


if kappa == None :
    kappa = cohen_kappa_score(y_validation, y_pred)
print("\nThe function Cohen Kappa Score is", kappa)


# In[91]:


if rauc == None :
    rauc = roc_auc_score(y_validation, y_pred)
print("\nArea Under the Receiver Operating Characteristic Curve (ROC AUC) is", rauc)


# In[92]:


if precision == None :
    precision = precision_score(y_validation, y_pred)
print("\nPrecision is",precision)


# In[93]:


if fscore == None :
    fscore = f1_score(y_validation, y_pred)
print("\nF1-Score is",fscore)


# In[94]:


if matrix.all() == None:
    matrix = confusion_matrix(y_validation, y_pred)
print("\nConfusion Matrix is \n\n\n",matrix)


# In[95]:


if report == None:
    report = classification_report(y_validation, y_pred)
print("\nClassification Report is \n\n\n",report)


# In[96]:


plotConfusionMatrix(matrix, target_names=['Evadidos','Graduados'], normalize=False)


# # A Decision Tree with Depth Level 3 for Exibithion

# In[97]:


# Starting Decision Tree
clfDTpadraoDepth3 = T.DecisionTreeClassifier(max_depth=3)
clfDTpadraoDepth3 = clfDTpadraoDepth3.fit(X_train, y_train)
clfDTpadraoDepth3


# In[98]:


# Building visualization by Graph with Gini Classifier
dataClass = T.export_graphviz(clfDTpadraoDepth3, out_file=None, filled=True, rounded=True, proportion=True,
            special_characters=True, feature_names=nameFeatures, class_names=['Evadido','Formado'])  
graph = gvz.Source(dataClass)  
graph 


# # Evaluate Classification Report

# In[99]:


print(classification_report(y_validation, y_pred))


# In[100]:


#confMatrix = ConfusionMatrix(y_validation, y_pred)
#confMatrix


# In[101]:


#confMatrix.print_stats()


# In[102]:


plotConfusionMatrix(confusion_matrix(y_validation, y_pred), normalize=False,target_names=['Evadidos','Graduados'])


# In[103]:


print("--- %s seconds ---" % (time.time() - starTime))