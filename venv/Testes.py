import unittest
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class First_Test(unittest.TestCase):

    # O que deverá realizar antes de efetuar qualquer teste
    # Inclui a criação das dataframes
    def setUp(self):
        self.df = pd.read_csv(
            '/home/anasapata/Personal/ProjetoIntegrado/Uso-de-Machine-Learning-para-previs-o-de-doen-as/breast-cancer-wisconsin.data.csv',
            header=None)
        self.df.columns = ['sample_code_number',
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

        robjects.r(
            'bc_data <- read.table("/home/anasapata/Personal/ProjetoIntegrado/Uso-de-Machine-Learning-para-previs-o-de-doen-as/breast-cancer-wisconsin.data.csv",'
            'header=FALSE, sep=",")')
        robjects.r('colnames(bc_data) <- c("sample_code_number",'
                   '"clump_thickness",'
                   '"uniformity_of_cell_size",'
                   '"uniformity_of_cell_shape",'
                   '"marginal_adhesion",'
                   '"single_epithelial_cell_size",'
                   '"bare_nuclei",'
                   '"bland_chromatin",'
                   '"normal_nucleoli",'
                   '"mitosis",'
                   '"classes")')

        self.df.replace('?', np.NaN, inplace=True)
        robjects.r('bc_data[bc_data == "?"] <- NA')

        self.df.iloc[:, 1:10] = self.df.iloc[:, 1:10].apply(lambda x: pd.to_numeric(x), 1)
        robjects.r('bc_data[,2:10] <- apply(bc_data[, 2:10], 2, function(x) as.numeric(as.character(x)))')

        # Irá ser utiliza a dataframe gerada pelo mice do R devido às diferenças nos algoritmos de modo a testar-se o restante
        robjects.r('library(mice)')
        robjects.r('dataset_impute <- mice(bc_data[, 2:10], print=FALSE)')
        robjects.r('bc_data_final <- cbind(bc_data[, 11, drop = FALSE], mice::complete(dataset_impute, 1))')
        robjects.r('bc_data_final$classes <- as.factor(bc_data$classes)')
        self.r_impute = robjects.r('bc_data_final')



    # Teste à alteração dos valores da variavel 'classes0 de maligno e bengino para 4 e 2 respetivamente
    def test_change_classes(self):

        self.df.classes.replace([2, 4], ['benign', 'malignant'], inplace=True)

        robjects.r('bc_data$classes <- ifelse(bc_data$classes == "2", "benign",'
                   'ifelse(bc_data$classes == "4", "malignant", NA))')

        for i in range(self.df.shape[0]):
            self.assertEqual(self.df['classes'][i],robjects.r('bc_data$classes')[i])

    # Teste à contagem de valores NA
    def test_count_na (self):
        null_columns = self.df.columns[self.df.isnull().any()]
        sum_na = self.df[null_columns].isnull().sum()
        self.assertEqual(int(sum_na), robjects.r('as.integer(length(which(is.na(bc_data))))')[0])

    # Teste à contagem de linha da dataframe
    def test_nrow(self):
        nrow = self.df.shape[0]
        self.assertEqual(nrow, robjects.r('nrow(bc_data)')[0])

    # Teste à contagem dos casos malignos e benignos
    def test_count_mal_ben(self):
        bening_cases_r = robjects.r('summary(bc_data_final$classes)[[1]]')
        malignant_cases_r = robjects.r('summary(bc_data_final$classes)[[2]]')
        r_results = [bening_cases_r[0], malignant_cases_r[0]]
        ben = self.r_impute[self.r_impute.classes == '2']
        mal = self.r_impute[self.r_impute.classes == '4']
        p_results = [ben.shape[0], mal.shape[0]]
        self.assertEqual(p_results, r_results)

    # Teste à PCA
    def test_pca(self):
        # PCA no r
        robjects.r('library("pcaGoPromoter")')
        robjects.r('library(ellipse)')
        robjects.r('pcaOutput <- pca(t(bc_data_final[, -1]), printDropped = FALSE, scale = TRUE, center = TRUE)')
        robjects.r('pcaOutput2 <- as.data.frame(pcaOutput$scores)')
        robjects.r('pcaOutput2$groups < - bc_data_final$classes')
        r_pca = robjects.r('pcaOutput2')
        # PCA python
        # Necessário obter os dados sem a primeira coluna e fazer a sua transposta
        df_without_classes = self.r_impute.iloc[:, 1:]
        df_without_classes_transpose = df_without_classes.transpose()
        df_normalize = StandardScaler().fit_transform(df_without_classes_transpose)
        # Irão ser encontradas duas componentes principais (9)
        pca_9 = PCA(n_components=9)
        principalComponents_9 = pca_9.fit_transform(df_normalize.transpose())

        # Aplicando PCA aos dados
        principalComponents_Df = pd.DataFrame(data=principalComponents_9,
                                              columns=['PC1',
                                                       'PC2',
                                                       'PC3',
                                                       'PC4',
                                                       'PC5',
                                                       'PC6',
                                                       'PC7',
                                                       'PC8',
                                                       'PC9'])
        principalComponents_Df['Group'] = self.r_impute['classes']
        print(principalComponents_Df['PC1'])
        print(r_pca['PC1'])
        self.assertEqual(principalComponents_Df['PC1'], r_pca['PC1'])


if __name__ == '__main__':
    unittest.main()