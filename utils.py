import pandas as pd
def stratified_sample(df, strata, size=None, seed=None, keep_index= True):
    '''
    It samples data from a pandas dataframe using strata. These functions use
    proportionate stratification:
    n1 = (N1/N) * n
    where:
        - n1 is the sample size of stratum 1
        - N1 is the population size of stratum 1
        - N is the total population size
        - n is the sampling size
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    :seed: sampling seed
    :keep_index: if True, it keeps a column with the original population index indicator
    
    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    Examples
    --------
    >> df.head()
    	id  sex age city 
    0	123 M   20  XYZ
    1	456 M   25  XYZ
    2	789 M   21  YZX
    3	987 F   40  ZXY
    4	654 M   45  ZXY
    ...
    # This returns a sample stratified by sex and city containing 30% of the size of
    # the original data
    >> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
    Requirements
    ------------
    - pandas
    - numpy
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)

    # controlling variable to create the dataframe or append to it
    first = True 
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry=''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum]
            n = tmp_grpd.iloc[i]['samp_size']

            if type(value) == str:
                value = "'" + str(value) + "'"
            
            if s != len(strata)-1:
                qry = qry + stratum + ' == ' + str(value) +' & '
            else:
                qry = qry + stratum + ' == ' + str(value)
        
        # final dataframe
        if first:
            stratified_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            first = False
        else:
            tmp_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)
    
    return stratified_df



def stratified_sample_report(df, strata, size=None):
    '''
    Generates a dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Returns
    -------
    A dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)
    return tmp_grpd


def __smpl_size(population, size):
    '''
    A function to compute the sample size. If not informed, a sampling 
    size will be calculated using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Parameters
    ----------
        :population: population size
        :size: sample size (default = None)
    Returns
    -------
    Calculated sample size to be used in the functions:
        - stratified_sample
        - stratified_sample_report
    '''
    if size is None:
        cochran_n = round(((1.96)**2 * 0.5 * 0.5)/ 0.02**2)
        n = round(cochran_n/(1+((cochran_n -1) /population)))
    elif size >= 0 and size < 1:
        n = round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n
## funcion de mapeo de variables
def mapeo_de_variables(df : pd.DataFrame) -> pd.DataFrame:
    """genera una tabla donde muestra, para cada variable en un dataframe, su número de nulos, tipo, valores únicos y porcentaje de nulos"""
    dimension=df.shape
    variables = df.columns.to_list()
    nulos = []
    variable = []
    tipo_variable = []
    valores_unicos = []
    unicos = []
    for variable in variables:
        nulos.append(df[variable].isnull().sum())
        tipo_variable.append(df[variable].dtype)
        valores_unicos.append(len(df[variable].dropna().unique()))
        unicos.append(df[variable].dropna().unique().tolist())
    tabla = pd.DataFrame({"Variable":variables,"Nulos":nulos,"Tipo Variable":tipo_variable,"Valores Unicos": valores_unicos,"Unicos":unicos})
    tabla["Porcentaje Nulos"]=(tabla["Nulos"]/len(df))*100
    tabla.sort_values("Porcentaje Nulos",ascending=False, inplace = True)
    return tabla,dimension

def identifica_la_lista_de_variables_constantes_y_las_elimina(df : pd.DataFrame, tabla_de_variables : pd.DataFrame) -> pd.DataFrame:
    """Elimina las columnas de un dataframe que tienen un único valor y retorna el nombre de esas columnas"""
    las_variables_constantes = tabla_de_variables[tabla_de_variables["Valores Unicos"] <= 1]["Variable"]
    df1=df.drop(las_variables_constantes.to_list(), axis=1, inplace = False)
    return df1,las_variables_constantes

def identifica_la_lista_de_variables_con_altos_nulos(df : pd.DataFrame, tabla_de_variables : pd.DataFrame, las_variables_con_valores_unicos : pd.DataFrame,por=80) -> pd.DataFrame:
    """Elimina las columnas de un dataframe que tienen mas de 80% valores nulos --y que no sean constantes-- y retorna el nombre de esas columnas"""
    las_variables_con_muchos_nulos = tabla_de_variables[tabla_de_variables["Porcentaje Nulos"] >= por]["Variable"]
    las_variables_con_muchos_nulos = list(set(las_variables_con_muchos_nulos.values) - set(las_variables_con_valores_unicos .values)) 
    df1=df.drop(las_variables_con_muchos_nulos, axis=1, inplace = False)
    return df1,las_variables_con_muchos_nulos


def recodificacion_variables(df,df_tmp):
    
# Para Variables Numericas, Valores Unicos hasta 5 la vamos a llamar tipo categórica, recodificamos como category
    variables_cat = df_tmp[df_tmp["Valores Unicos"] <= 5]["Variable"]
    for variable in list(variables_cat):
        df[variable]=df[variable].astype('category')    
        
# Recodificar todo lo string a category. Revisar antes de recodificar
    tabla_info_df_final= mapeo_de_variables(df)
    for variable in list(df.dtypes[df.dtypes == 'object'].index):
        df[variable]=df[variable].astype('category')
        
    return df


def conteo_tipos_variable(df):
    tmp_for_plot=pd.DataFrame(df.dtypes.value_counts())
    tmp_for_plot.reset_index(inplace=True)
    tmp_for_plot['index']=tmp_for_plot['index'].astype('string')
    tabla=tmp_for_plot.groupby(['index']).sum().sort_values(0,ascending=False)
    return(tabla)

def type_cols(df):  
    numeric_cols=list(df.select_dtypes(include='number').columns)
    cat_cols=list(df.dtypes[df.dtypes == 'category'].index)
    return (numeric_cols,cat_cols)



def remove_collinear_features(x, threshold):
    '''
    Objetivo:
        Eliminar características colineales en un marco de datos con un coeficiente de correlación
        mayor que el umbral. La eliminación de características colineales puede ayudar a un modelo
        generalizar y mejora la interpretabilidad del modelo.

    Entradas:
        x: marco de datos de características
        umbral: se eliminan las entidades con correlaciones superiores a este valor

    Producción:
        marco de datos que contiene solo las características no altamente colineales
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)

    return x

def TablaX(df,VariablesNumericas,VariablesCategoricas):
    datos=df
    if VariablesCategoricas != [] :
        datos_dummies=pd.get_dummies(datos[VariablesCategoricas],drop_first=True)
        X=pd.concat([datos_dummies,datos[VariablesNumericas]],axis=1,sort=False)
    else:
        X=datos[VariablesNumericas].apply
    return X    


## Funcion correlacion X numericas con Y

def correlacion_x_y(df, pred=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    uniques=uniques.iloc[0,:]
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 

    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis = 1)
    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    #print('___________________________\nData types:\n',str.types.value_counts())
    #print('___________________________')
    return str