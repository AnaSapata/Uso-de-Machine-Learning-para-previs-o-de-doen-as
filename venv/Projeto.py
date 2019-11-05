#importação da biblioteca pandas, quando for necessário usar a mesma irá ser utilizado pd em vez de pandas
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import elipse_confidence_plot


# Leitura do ficheiro dos dados, especificando que o mesmo não tem nome para as colunas (header = None)
# Comando read_csv da biblioteca Pandas é o equivalente ao read.table do R, uma vez que temos o ficheiro em formato csvx-special/nautilus-clipboard

#df = pd.read_csv('/home/raquel/Uso-de-Machine-Learning-para-previs-o-de-doen-as/breast-cancer-wisconsin.data.csv',
#                 header = None)
df = pd.read_csv('/home/anasapata/Personal/ProjetoIntegrado/Uso-de-Machine-Learning-para-previs-o-de-doen-as/breast-cancer-wisconsin.data.csv',
                 header = None)

# Mostra as primeiras 5 linhas do ficheiro/data frame
# print(df.head())

# Uma vez que o ficheiro não tem nome para as colunas, tal como acontece posteriormente com a data frame
# é então necessário atribuir os respetivos nomes às mesmas para tal
df.columns = ['sample_code_number',
              'clump_thickness',
              'uniformity_of_cell_size',
              'uniformity_of_cell_shape',
              'marginal_adhesion',
              'single_epithelial_cell_size',
              'bare_nuclei',
              'bland_chromatin',
              'normal_nucleoli',
              'mitosis',
              'classes']

# Mostrar novamente as primeira 5 linhas de modo a confirmar que os nomes das colunas lhes foram atru«ibuidos
# print(df.head())

# Quando classes tem o valor 2 deverá torna-se "benign", quando tem o valor 4 deverá tornar-se "malignant" e nos restantes casos NA
df.classes.replace([2, 4], ['benign', 'malignant'], inplace = True)

# Verificar que alterou os valores
# print(df.head())
# print(df.tail())

# Quando existe o valor ? é atribuido ao mesmo o valor NaN (equivalente ao NA)
df.replace('?', np.NaN, inplace = True)

# Verifica quais as colunas com valores nulos
null_columns = df.columns[df.isnull().any()]

# Conta o número de celulas com valores nulos
print("Numero de celulas sem valor")
print(df[null_columns].isnull().sum())

# Verificar o tipo dos elementos das colunas 1:10 antes de proceder à alteração
# print(df.dtypes)

# Passar os elementos das colunas 2:10 para o tipo numerico
df.iloc[:,1:10] = df.iloc[:,1:10].apply(lambda x: pd.to_numeric(x),1)

# Verificar se os elementos das colunas referidas já se encontram todos em formato numerico
# print(df.dtypes)


# https://scikit-learn.org/stable/modules/impute.html
# https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py
# https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html#sphx-glr-auto-examples-impute-plot-iterative-imputer-variants-comparison-py
# Informar de como deverá ser feito o impute dos dados
imp = SimpleImputer(missing_values = np.NaN, strategy = 'mean')
# Verificar o que está a ser aplicado
imp.fit(df.iloc[:,1:10])

# Realizar a transformação dos dados
df_impute = imp.transform(df.iloc[:,1:10])
# Uma vez que o df_impute é do tipo numpy.ndarray é utilizado o metodo savetxt do numpy para
# guardar os resultados obtidos e verificar que já não existem NaN
# np.savetxt('/home/raquel/teste.csv', df_impute, delimiter = ";")
# np.savetxt('/home/anasapata/Personal/ProjetoIntegrado/teste.csv', df_impute, delimiter = ";")

# Utilizar outro metodo para impute
imp2 = IterativeImputer(estimator = KNeighborsRegressor(n_neighbors = 15), random_state = 0)
imp2.fit(df.iloc[:,1:10])
df_impute2 = imp2.transform(df.iloc[:,1:10])
# np.savetxt('/home/raquel/teste_2_2.csv', df_impute2, delimiter = ";")
# np.savetxt('/home/anasapata/Personal/ProjetoIntegrado/teste_2.csv', df_impute2, delimiter = ";")

# Como o resultado do impute é um numpy ndarray existe a necessidade de passar o mesmo para o formato data frame
df_impute2_df = pd.DataFrame(data = df_impute2)

# atribuir o nome das colunas aos dados onde foi resultado o impute
df_impute2_df.columns = ['clump_thickness',
              'uniformity_of_cell_size',
              'uniformity_of_cell_shape',
              'marginal_adhesion',
              'single_epithelial_cell_size',
              'bare_nuclei',
              'bland_chromatin',
              'normal_nucleoli',
              'mitosis']

# Converter todas as colunas para inteiro
df_impute2_df = df_impute2_df.astype('int64')

# Selecionar a ultima coluna do data frame original para se poder efetuar o merge com os dados com o impute
cf = df.iloc[:,10]

# Colocar todas as data frames a juntar num array
L = [cf, df_impute2_df]
# Fazer o merge de todos os dados
df_final = pd.concat(L, axis = 1)

# Definir a coluna classes como uma variavel categorica
df_final['classes'] = df_final['classes'].astype('category')

ben = df_final[df_final.classes == 'benign']
mal = df_final[df_final.classes == 'malignant']
summary_classes = 'benign     malignant\n' + str(ben.shape[0]) + '        ' + str(mal.shape[0])
print(summary_classes)

# Histograma
print("Cancer data set dimensions : {}".format(df.shape)) #Dimensão de Conjunto de dados

# Obtenção da variavel classes
dados=df_final['classes']

x = np.arange(2)
colors = ['green', 'red']
plt.bar(x, height= [ben.shape[0],mal.shape[0]], color=colors )
plt.xticks(x, ['benign','malignant'])
plt.xlabel('classes')
plt.ylabel('count')
plt.title('Prevenção de Doenças')
plt.show()

#Response variable for regression
# Grafico- http://pandas-docs.github.io/pandas-docs-travis/user_guide/visualization.html In[35],In[36]

df_final['clump_thickness'].hist( figsize=(6, 4),color='c',) #figsize-dimensao do grafico
plt.xlabel('clump_thickness')
plt.ylabel('count')
plt.show()



# Necessário obter os dados sem a primeira coluna e fazer a sua transposta
df_without_classes = df_final.iloc[:,1:]
df_without_classes_transpose = df_without_classes.transpose()
df_normalize = StandardScaler().fit_transform(df_without_classes_transpose)
# Confirmação da normalização dos dados
# print("(" + str(np.mean(df_normalize)) +","+str(np.std(df_normalize))+")")

# Irão ser encontradas duas componentes principais (9)
pca_9 = PCA(n_components = 9)
# Aplicado PCA aos dados
principalComponents_9 = pca_9.fit_transform(df_normalize)

# Data Frame para observação do valor de cada variavel na respetiva componente
principal_Df_2 = pd.DataFrame(data = pca_9.components_[:,[0,1]]
             , columns = ['principal component 1', 'principal component 2'])
print(principal_Df_2)


principalComponents_Df = pd.DataFrame(data = pca_9.components_.transpose(),
              columns = ['PC1',
              'PC2',
              'PC3',
              'PC4',
              'PC5',
              'PC6',
              'PC7',
              'PC8',
              'PC9'])
principalComponents_Df['Group'] = df_final['classes']
print(principalComponents_Df.head())
# Percentagem de explicação de cada componente
print('Explained variation per principal component: {}'.format(pca_9.explained_variance_ratio_))
print('Vetores pp per principal component: {}'.format(pca_9.singular_values_))

fig, ax = plt.subplots()
plt.xlabel('PC1: ' + str(round(pca_9.explained_variance_ratio_[0],2)*100) + "% Variance")
plt.ylabel('PC2: ' + str(round(pca_9.explained_variance_ratio_[1],2)*100) + "% Variance")
plt.title("Principal Component Analysis of Breast Cancer Dataset")
targets = ['benign', 'malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = principalComponents_Df['Group'] == target
    plt.scatter(principalComponents_Df.loc[indicesToKeep, 'PC1']
               , principalComponents_Df.loc[indicesToKeep, 'PC2'], c = color, s = 50)
    elipse_confidence_plot.confidence_ellipse(principalComponents_Df.loc[indicesToKeep, 'PC1'], principalComponents_Df.loc[indicesToKeep, 'PC2'],
                       ax, alpha=0.5, facecolor=color, edgecolor=color, zorder=0)

plt.legend(targets,prop={'size': 10})
plt.show()

#------------------Treinamento, validacao e teste dos dados-------------------------------------------- ?

random.seed(42)

#createDataPartition


training_set, testing_set= train_test_split(df_final,test_size=0.7)

#--------------------------Graficos GGPLOT---------------------------
#Useful functions 
#def labels(from_, to_, step_):
#   return pd.Series(np.arange(from_, to_ + step_, step_)).apply(lambda x: '{:,}'.format(x)).tolist()def breaks(from_, to_, step_):
#    return pd.Series(np.arange(from_, to_ + step_, step_)).tolist()

#Read data and set index

#data = pd.read_csv(c_remote_data)data = (
 #   data.filter([
  #      'abv',
   #     'ibu',
    #    'id',
     #   'name',
      #  'style',
       # 'brewery_id',
        ##'ounces'
 #   ]).
  #  set_index('id')
#)
#
#Histogram
#Initial

#fig = (
 #   ggplot(data.dropna(subset = ['abv'])) +
  #  geom_histogram(aes(x = 'abv'))
#)

#--------------------------------------------------------------------//





