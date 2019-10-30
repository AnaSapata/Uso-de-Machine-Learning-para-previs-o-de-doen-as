import unittest
import pandas as pd
import rpy2.robjects as robjects

class TestStringMethods(unittest.TestCase):

    def test(self):
        df = pd.read_csv(
            '/home/anasapata/Personal/ProjetoIntegrado/Uso-de-Machine-Learning-para-previs-o-de-doen-as/breast-cancer-wisconsin.data.csv',
            header=None)
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

        df.classes.replace([2, 4], ['benign', 'malignant'], inplace=True)

        robjects.r('bc_data$classes <- ifelse(bc_data$classes == "2", "benign",'
                   'ifelse(bc_data$classes == "4", "malignant", NA))')

        for i in range(df.shape[0]):
            self.assertEqual(df['classes'][i],robjects.r('bc_data$classes')[i])

if __name__ == '__main__':
    unittest.main()