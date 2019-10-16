#importação da biblioteca pandas, quando for necessário usar a mesma irá ser utilizado pd em vez de pandas
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# Leitura do ficheiro dos dados, especificando que o mesmo não tem nome para as colunas (header = None)
# Comando read_csv da biblioteca Pandas é o equivalente ao read.table do R, uma vez que temos o ficheiro em formato csv
df = pd.read_csv('/home/anasapata/Personal/ProjetoIntegrado/Uso-de-Machine-Learning-para-previs-o-de-doen-as/breast-cancer-wisconsin.data.csv',
                 header = None)

# Mostra as primeiras 5 linhas do ficheiro/data frame
print(df.head())

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
print(df.head())

# Quando classes tem o valor 2 deverá torna-se "benign", quando tem o valor 4 deverá tornar-se "malignant" e nos restantes casos NA
df.classes.replace([2, 4], ['benign', 'malignant'], inplace = True)

# Verificar que alterou os valores
print(df.head())
print(df.tail())

# Quando existe o valor ? é atribuido ao mesmo o valor NaN (equivalente ao NA)
df.replace('?', np.NaN, inplace = True)

# Verifica quais as colunas com valores nulos
null_columns = df.columns[df.isnull().any()]

# Conta o número de celulas com valores nulos
print(df[null_columns].isnull().sum())

# Verificar o tipo dos elementos das colunas 2:10 antes de proceder à alteração
print(df.dtypes)

# Passar os elementos das colunas 2:10 para o tipo numerico
df.iloc[:,1:9] = df.iloc[:,1:9].apply(lambda x: pd.to_numeric(x),1)

# Verificar se os elementos das colunas referidas já se encontram todos em formato numerico
print(df.dtypes)


# https://scikit-learn.org/stable/modules/impute.html
# Informar de como deverá ser feito o impute dos dados
imp = SimpleImputer(missing_values = np.NaN, strategy = 'mean')
# Verificar o que está a ser aplicado
imp.fit(df.iloc[:,1:9])

# Realizar a transformação dos dados
df_impute = imp.transform(df.iloc[:,1:9])
# Uma vez que o df_impute é do tipo numpy.ndarray é utilizado o metodo savetxt do numpy para
# guardar os resultados obtidos e verificar que já não existem NaN
np.savetxt('/home/anasapata/Personal/ProjetoIntegrado/teste.csv', df_impute, delimiter = ";")

# Utilizar outro mmetodo para impute
imp2 = IterativeImputer(max_iter = 10, random_state = 0)
imp2.fit(df.iloc[:,1:9])
df_impute2 = imp2.transform(df.iloc[:,1:9])
np.savetxt('/home/anasapata/Personal/ProjetoIntegrado/teste_2.csv', df_impute2, delimiter = ";")

#df_impute2['classes'] = df_impute2['classes'].astype('category')
#print( 'Aqui')
#print(df_impute2)
#pd.concat(df.iloc[:,0], df_impute2, df.iloc[:,10])
#ben = df_impute2[df_impute2.classes == 'bening']
#print(ben.shape)

