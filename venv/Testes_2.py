import unittest
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from autoimpute.imputations import SingleImputer, MultipleImputer

class MyTestCase(unittest.TestCase):

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

    def test_impute(self):
        si = SingleImputer(strategy={'bare_nuclei':"pmm"})
        df_impute = si.fit_transform(self.df.iloc[:,1:10])
        robjects.r('library(mice)')
        robjects.r('dataset_impute <- mice(bc_data[, 2:10], print=FALSE)')
        r_impute = robjects.r('mice::complete(dataset_impute,1)$bare_nuclei')
        for i in range(df_impute.shape[0]):
            self.assertEqual(df_impute['bare_nuclei'][i],r_impute[i])


if __name__ == '__main__':
    unittest.main()
