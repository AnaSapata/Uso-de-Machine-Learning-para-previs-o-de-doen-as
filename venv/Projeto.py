#importação da biblioteca pandas, quando for necessário usar a mesma irá ser utilizado pd em vez de pandas
import pandas as pd
import numpy as np

# Leitura do ficheiro dos dados, especificando que o mesmo não tem nome para as colunas (header = None)
# Comando read_csv da biblioteca Pandas é o equivalente ao read.table do R, uma vez que temos o ficheiro em formato csv
df = pd.read_csv('/home/jsm/Uso-de-Machine-Learning-para-previs-o-de-doen-as/breast-cancer-wisconsin.data.csv',
                 header = None)
#Tenho duvida

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






# DADOS A FALTAR DO INPUT /////////////////////////////////////////////////////////

# Cod em Python ---------------- bc_data[,2:10] <- apply(bc_data[, 2:10],2, function(x) as.numeric(as.character(x)))

df.loc[,2:10)] = df.loc[,2:10)].apply(lambda x: pd.to_numeric(x),1)

# HELP                dataset_impute <- mice(bc_data[, 2:10],  print = FALSE)

# HELP                bc_data <- cbind(bc_data[, 11, drop = FALSE],
# HELP                mice::complete(dataset_impute, 1))

# Cod em Python ---------------- bc_data$classes <-as.factor(bc_data$classes)
df.classes = df.astype(df.classes) 

# QUAntos casos benignos e malignos existem ?

# Cod em Python ---------------- summary(bc_data$classes)

summary = df.describe()
summary = summary.transpose()
summary.head()
