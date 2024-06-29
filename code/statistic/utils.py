import pandas as pd
import sklearn
import numpy as np
import rsatoolbox
def calc_rdm(arr1,label1,arr2 = None,label2 = None,errmin =0.1,errmax=0.8):
    arr1 = pd.Series(arr1)
    level_1q = arr1.quantile(errmin)
    level_3q = arr1.quantile(errmax)

    arr1 = np.where(arr1>level_3q, level_3q, arr1)
    arr1 = np.where(arr1<level_1q, level_1q, arr1)
    gdf = pd.DataFrame(arr1)
    gdf['target']= label1
    gdf = gdf.sort_values(by = ['target'])

    if(type(arr2) != np.ndarray):
        fdf = gdf
    else:
        arr2 = pd.Series(arr2)
        level_1q = arr2.quantile(errmin)
        level_3q = arr2.quantile(errmax)
        arr2 = np.where(arr2>level_3q, level_3q, arr2)
        arr2 = np.where(arr2<level_1q, level_1q, arr2)

        edf = pd.DataFrame(arr2)
        edf['target']= label2
        edf = edf.sort_values(by = ['target'])
        fdf = pd.concat([gdf,edf])

    fnp = np.array(fdf.iloc[:,:1 ])
    data = rsatoolbox.data.Dataset(fnp)
    rdms = rsatoolbox.rdm.calc_rdm(data)
    return rdms

def calc_rdm_emotion(arr1,label1,arr2 = None,label2 = None,errmin =0.1,errmax=0.8):
    arr1 = pd.Series(arr1)
    level_1q = arr1.quantile(errmin)
    level_3q = arr1.quantile(errmax)

    arr1 = np.where(arr1>level_3q, level_3q, arr1)
    arr1 = np.where(arr1<level_1q, level_1q, arr1)
    gdf = pd.DataFrame(arr1)
    gdf['target']= label1
    

    if(type(arr2) != np.ndarray):
        fdf = gdf
    else:
        arr2 = pd.Series(arr2)
        level_1q = arr2.quantile(errmin)
        level_3q = arr2.quantile(errmax)
        arr2 = np.where(arr2>level_3q, level_3q, arr2)
        arr2 = np.where(arr2<level_1q, level_1q, arr2)
        
        edf = pd.DataFrame(arr2)
        edf['target']= label2
        fdf = pd.concat([gdf,edf])
    fdf = fdf.sort_values(by = ['target'])
    fnp = np.array(fdf.iloc[:,:1 ])
    data = rsatoolbox.data.Dataset(fnp)
    rdms = rsatoolbox.rdm.calc_rdm(data)
    return rdms

def anova_test(arr,method = "normal"):
    import scipy.stats
    from statsmodels.stats.anova import AnovaRM
    if(method == "normal"):
        group_list = []
        value_list = arr.transpose().reshape(-1)
        for i in range (0,7):
            group_list += [i for j in range(0,15)]
        sub_list=[]
        for i in range (0,7):
            sub_list += [j for j in range(0,15)]
        df = pd.DataFrame({'Group':group_list, 'id' :sub_list,'Value':value_list})
        return(AnovaRM(data=df, depvar='Value', subject='id', within=['Group']).fit())
    else:
        group_list = []
        value_list = arr.transpose().reshape(-1)
        for i in range (0,7):
            group_list += [i for j in range(0,15)]
        df = pd.DataFrame({'Group':group_list,'Value':value_list})

        from statsmodels.sandbox.stats.multicomp import MultiComparison
        from scipy.stats import ttest_ind
        
        comp = MultiComparison(df['Value'], df['Group'])
        ret = comp.allpairtest(scipy.stats.ttest_rel, method='bonf')
        return ret[0]

def anova_test_2way(arr1,arr2,method = 'normal', factor=1 ):
    if(method == "normal"):
        import scipy.stats
        level_list = []
        group_list = []
        value_list = np.concatenate([arr1.transpose().reshape(-1),arr2.transpose().reshape(-1)],axis=0)
        for i in range (0,7):
            group_list += [i for j in range(0,15)]
            level_list += ["G" for j in range(0,15)]
        for i in range (0,7):
            group_list += [i for j in range(0,15)]
            level_list += ["E" for j in range(0,15)]
        
        df = pd.DataFrame({'Level':level_list,'Group':group_list,'Value':value_list})

        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        #perform two-way ANOVA
        model = ols('Value ~ C(Level) + C(Group) + C(Level):C(Group)', data=df).fit()
        return sm.stats.anova_lm(model, typ=2)
    else:
        from bioinfokit.analys import stat
        # perform multiple pairwise comparison (Tukey HSD)
        # unequal sample size data, tukey_hsd uses Tukey-Kramer test
        level_list = []
        group_list = []
        value_list = np.concatenate([arr1.transpose().reshape(-1),arr2.transpose().reshape(-1)],axis=0)
        for i in range (0,7):
            group_list += [i for j in range(0,15)]
            level_list += ["G" for j in range(0,15)]
        for i in range (0,7):
            group_list += [i for j in range(0,15)]
            level_list += ["E" for j in range(0,15)]

        df = pd.DataFrame({'Level':level_list,'Group':group_list,'Value':value_list})
        res = stat()
        # for main effect Genotype
        if(factor ==1):
            xfac_var_ = "Group"
        else:
            xfac_var_ = "Level"
        res.tukey_hsd(df=df, res_var='Value', xfac_var=xfac_var_, anova_model='Value~C(Level)+C(Group)+C(Level):C(Group)')
        return(res.tukey_summary)

def convert_df(arr1):

    people_list = []
    value_list = []
    for element in arr1:
        value_list += element.transpose().tolist()
    
    for i in range (0,15):
        people_list += [i for j in range(0,20)]

    value_list= np.array(value_list)

    df = pd.concat([pd.DataFrame(people_list),pd.DataFrame(value_list)],axis=1)
    df.columns = ["People","Ha","Sa","Su","An","Di","Fe","Ne"]
    return df 