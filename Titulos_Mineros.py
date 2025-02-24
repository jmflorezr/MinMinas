# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 20:01:31 2025
Instalar las librerias requeridas.

@author: JULIAN FLOREZ
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LECTURA DE SHAPEFILES CON GEOPANDAS
# ------------------------------------------------------------------
titulo_gdf = gpd.read_file(r"E:\MinMinas\Pruebas\Titulo_Vigente\Título Vigente.shp")
subcontrato_gdf = gpd.read_file(r"E:\MinMinas\Pruebas\Subcontrato\Subcontrato.shp")
solicitud_gdf = gpd.read_file(r"E:\MinMinas\Pruebas\Solicitud_Vigente\Solicitud Vigente.shp")

# 2. CONVERTIR A DATAFRAME (ELIMINANDO LA COLUMNA 'geometry')
# ------------------------------------------------------------------
titulo_df = pd.DataFrame(titulo_gdf.drop(columns='geometry', errors='ignore'))
subcontrato_df = pd.DataFrame(subcontrato_gdf.drop(columns='geometry', errors='ignore'))
solicitud_df = pd.DataFrame(solicitud_gdf.drop(columns='geometry', errors='ignore'))

# 3. DEFINIR UNA FUNCIÓN DE EDA BÁSICA
# ------------------------------------------------------------------
def quick_eda(df, name="DataFrame"):
    """
    Realiza un EDA básico con:
      - Info del DataFrame (tipos de dato, etc.)
      - Primeras filas
      - Estadísticos descriptivos de columnas numéricas
      - Conteo de valores nulos
      - Histogramas (AREA_HA) y countplots (CLASIFICAC, TENURE_STA, ACTIVE_TEN) si existen
      - Ejemplo de agrupación por DEPARTAMEN para ver áreas totales
    """
    print(f"\n=== {name} ===")
    
    # Info general
    print("\n[INFO]")
    print(df.info())
    
    # Primeras filas
    print("\n[HEAD]")
    print(df.head())
    
    # Estadísticos básicos numéricos
    print("\n[DESCRIBE - Numérico(Hs)]")
    print(df.describe(include=[float]))
    
    # Valores nulos
    print("\n[VALORES NULOS]")
    print(df.isnull().sum())
    
    # Histograma de AREA_HA (si existe)
    if 'AREA_HA' in df.columns:
        plt.figure(figsize=(7,5))
        sns.histplot(df['AREA_HA'].dropna(), bins=20, kde=False)
        plt.title(f"Distribución de AREA_HA - {name}")
        plt.xlabel("Área (ha)")
        plt.ylabel("Frecuencia")
        plt.show()
    
    # Conteo de CLASIFICAC (si existe)
    if 'CLASIFICAC' in df.columns:
        plt.figure(figsize=(7,5))
        sns.countplot(x='CLASIFICAC', data=df, order=df['CLASIFICAC'].value_counts().index)
        plt.title(f"Distribución de CLASIFICAC - {name}")
        plt.xlabel("Clasificación")
        plt.ylabel("Conteo")
        plt.xticks(rotation=45)
        plt.show()

    # Conteo de TENURE_STA (si existe)
    if 'TENURE_STA' in df.columns:
        plt.figure(figsize=(7,5))
        sns.countplot(x='TENURE_STA', data=df, order=df['TENURE_STA'].value_counts().index)
        plt.title(f"Distribución de TENURE_STA - {name}")
        plt.xlabel("Estado de Título")
        plt.ylabel("Conteo")
        plt.xticks(rotation=45)
        plt.show()

    # Conteo de ACTIVE_TEN (si existe)
    if 'ACTIVE_TEN' in df.columns:
        plt.figure(figsize=(7,5))
        sns.countplot(x='ACTIVE_TEN', data=df, order=df['ACTIVE_TEN'].value_counts().index)
        plt.title(f"Distribución de ACTIVE_TEN - {name}")
        plt.xlabel("¿Título Activo?")
        plt.ylabel("Conteo")
        plt.show()
    
    # Suma de áreas por departamento (si existen DEPARTAMEN y AREA_HA)
    if 'DEPARTAMEN' in df.columns and 'AREA_HA' in df.columns:
        area_por_depto = df.groupby('DEPARTAMEN')['AREA_HA'].sum().sort_values(ascending=False)
        print("\n[TOP 5 DEPARTAMENTOS POR ÁREA TOTAL(Hs)]")
        print(area_por_depto.head(5))

# 4. EJECUTAR EL EDA BÁSICO PARA CADA DATAFRAME
# ------------------------------------------------------------------
quick_eda(titulo_df, "Título Vigente")
quick_eda(subcontrato_df, "Subcontrato")
quick_eda(solicitud_df, "Solicitud Vigente")

    
###############################################################
unique_values = titulo_df['TIPO_TERMI'].unique()
unique_values = [x for x in titulo_df['TIPO_TERMI'].unique() if x is not None]

# 4.1. Crear el nuevo dataframe "titulo_df02" filtrando filas donde "TIPO_TERMI" sea vacío
titulo_df02 = titulo_df[~titulo_df['TIPO_TERMI'].isin(unique_values)]

# 4.2. Convertir "FECHA_DE_S" a datetime y extraer el año
titulo_df02['FECHA_DE_S'] = pd.to_datetime(titulo_df02['FECHA_DE_S'], errors='coerce')
titulo_df02['year'] = titulo_df02['FECHA_DE_S'].dt.year

def plot_grouped_bar(data, group_var, top_n_per_year=None):
    """
    Esta función agrupa 'data' por 'year' y por la variable 'group_var',
    sumando la columna 'AREA_HA'. Si se especifica 'top_n_per_year', se
    filtra para conservar solo los n primeros registros por año (según la
    suma de AREA_HA). Luego, la función:
      1. Genera un diagrama de barras (eje X: año, barras diferenciadas según 'group_var')
         mostrando la suma de AREA_HA.
      2. Genera un gráfico de dispersión con los mismos ejes, pero mostrando los valores
         individuales de AREA_HA (sin agregación).
      3. Imprime la tabla resultante de la agregación.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # 1. Agrupar por año y la variable de interés sumando AREA_HA
    grouped = data.groupby(['year', group_var])['AREA_HA'].sum().reset_index()
    # Asegurarse de que AREA_HA sea numérico
    grouped['AREA_HA'] = pd.to_numeric(grouped['AREA_HA'], errors='coerce')
    
    # Filtrar para conservar solo los top n por año, si se especifica
    if top_n_per_year is not None:
        def top_n(df):
            return df.sort_values('AREA_HA', ascending=False).head(top_n_per_year)
        grouped = grouped.groupby('year', group_keys=False).apply(top_n)
    
    # Pivotear la tabla para graficar el diagrama de barras
    pivot_df = grouped.pivot(index='year', columns=group_var, values='AREA_HA').fillna(0)
    
    # 2. Diagrama de barras (suma de AREA_HA)
    pivot_df.plot(kind='bar', stacked=False, figsize=(10,6))
    plt.ylabel('Suma de AREA_HA')
    plt.title(f'Suma de AREA_HA por Año y {group_var}')
    plt.legend(title=group_var)
    plt.tight_layout()
    plt.show()
    
    # Imprimir la tabla resultante
    print(f"Tabla de Suma de AREA_HA por Año y {group_var}:")
    print(grouped.sort_values(['year', 'AREA_HA'], ascending=[True, False]))
    print("\n" + "="*80 + "\n")
    
    # 3. Gráfico de dispersión (valores individuales de AREA_HA)
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data, x='year', y='AREA_HA', hue=group_var)
    plt.ylabel('AREA_HA')
    plt.title(f'Gráfico de dispersión de AREA_HA por Año y {group_var}')
    plt.legend(title=group_var)
    plt.tight_layout()
    plt.show()


# Definir las columnas en las que se muestran todos los valores
cols_completas = ['PUBLICADO_', 'TITLE_TYPE', 'TENURE_STA', 'MODALIDAD','MINING_CLA', 'TITULO_EST', 'CLASIFICAC', 'ETAPA', 'PAR']

# Generar diagrama de barras y tabla para cada una de las variables completas
for col in cols_completas:
    plot_grouped_bar(titulo_df02, col)


# Lista de columnas a analizar
group_columns = ['SOLICITANT', 'DEPARTAMEN', 'MINERALES', 'MINERALES_']

for col in group_columns:
    # Agrupar por la columna y sumar AREA_HA
    grouped = titulo_df02.groupby(col)['AREA_HA'].sum().reset_index()
    
    # Ordenar la tabla de mayor a menor según AREA_HA
    grouped = grouped.sort_values('AREA_HA', ascending=False)
    
    # Tomar solo los 10 primeros
    grouped_top10 = grouped.head(10)
    
    # Mostrar la tabla resultante
    print(f"Tabla de suma de AREA_HA por {col} (Top 10):")
    print(grouped_top10)
    print("\n" + "="*50 + "\n")
    
    # Crear el diagrama de barras para los 10 primeros
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x=col, y='AREA_HA', data=grouped_top10)
    plt.title(f'Suma de AREA_HA por {col} (Top 10)')
    plt.xlabel(col)
    plt.ylabel('Suma de AREA_HA')
    
    # Personalizar las etiquetas del eje X:
    plt.xticks(rotation=90, fontsize=7, color='black')
    
    plt.tight_layout()
    plt.show()

##############################################################################

unique_values1 = subcontrato_df['TIPO_TERMI'].unique()
unique_values1 = [x for x in subcontrato_df['TIPO_TERMI'].unique() if x is not None]


# 4.1. Crear el nuevo dataframe "subcontrato_df02" filtrando filas donde "TIPO_TERMI" sea vacío
subcontrato_df02 = subcontrato_df[~subcontrato_df['TIPO_TERMI'].isin(unique_values1)]

# 4.2. Convertir "FECHA_DE_S" a datetime y extraer el año
subcontrato_df02['FECHA_DE_E'] = pd.to_datetime(subcontrato_df02['FECHA_DE_E'], errors='coerce')
subcontrato_df02['year'] = subcontrato_df02['FECHA_DE_E'].dt.year

def plot_grouped_bar(data, group_var, top_n_per_year=None):
    """
    Esta función realiza las siguientes operaciones:
      1. Agrupa 'data' por 'year' y por la variable 'group_var' y suma la columna 'AREA_HA'.
         Si se especifica 'top_n_per_year', se filtra para conservar solo los n primeros
         registros por año (según la suma de AREA_HA).
      2. Genera un diagrama de barras (eje X: año, barras diferenciadas según 'group_var')
         basado en la suma de AREA_HA.
      3. Imprime la tabla resultante de la agregación.
      4. Genera un gráfico de dispersión utilizando los datos individuales de AREA_HA,
         con eje X 'year' y eje Y 'AREA_HA', diferenciando los puntos según 'group_var'.
    """
    # 1. Agrupar por año y la variable de interés sumando AREA_HA
    grouped = data.groupby(['year', group_var])['AREA_HA'].sum().reset_index()
    # Asegurarse de que AREA_HA sea numérico
    grouped['AREA_HA'] = pd.to_numeric(grouped['AREA_HA'], errors='coerce')
    
    # Filtrar para conservar solo los top n registros por año, si se especifica
    if top_n_per_year is not None:
        def top_n(df):
            return df.sort_values('AREA_HA', ascending=False).head(top_n_per_year)
        grouped = grouped.groupby('year', group_keys=False).apply(top_n)
    
    # Pivotear la tabla para graficar el diagrama de barras
    pivot_df = grouped.pivot(index='year', columns=group_var, values='AREA_HA').fillna(0)
    
    # 2. Diagrama de barras (suma de AREA_HA)
    pivot_df.plot(kind='bar', stacked=False, figsize=(10,6))
    plt.ylabel('Suma de AREA_HA')
    plt.title(f'Suma de AREA_HA por Año y {group_var}')
    plt.legend(title=group_var)
    plt.tight_layout()
    plt.show()
    
    # 3. Imprimir la tabla resultante
    print(f"Tabla de Suma de AREA_HA por Año y {group_var}:")
    print(grouped.sort_values(['year', 'AREA_HA'], ascending=[True, False]))
    print("\n" + "="*80 + "\n")
    
    # 4. Gráfico de dispersión (datos individuales)
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data, x='year', y='AREA_HA', hue=group_var, palette="viridis")
    plt.ylabel('AREA_HA')
    plt.title(f'Gráfico de dispersión de AREA_HA por Año y {group_var}')
    plt.legend(title=group_var)
    plt.tight_layout()
    plt.show()


# Definir las columnas en las que se muestran todos los valores
cols_completas = ['PUBLICADO_', 'TENURE_STA', 'MINING_CLA', 'TITULO_EST', 'CLASIFICAC','ETAPA', 'PAR']

# Generar diagrama de barras y tabla para cada una de las variables completas
for col in cols_completas:
    plot_grouped_bar(subcontrato_df02, col)


# Lista de columnas a analizar
group_columns = ['SOLICITANT', 'DEPARTAMEN', 'MINERALES', 'MINERALES_']

for col in group_columns:
    # Agrupar por la columna y sumar AREA_HA
    grouped = subcontrato_df02.groupby(col)['AREA_HA'].sum().reset_index()
    
    # Ordenar la tabla de mayor a menor según AREA_HA
    grouped = grouped.sort_values('AREA_HA', ascending=False)
    
    # Tomar solo los 10 primeros
    grouped_top10 = grouped.head(10)
    
    # Mostrar la tabla resultante
    print(f"Tabla de suma de AREA_HA por {col} (Top 10):")
    print(grouped_top10)
    print("\n" + "="*50 + "\n")
    
    # Crear el diagrama de barras para los 10 primeros
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x=col, y='AREA_HA', data=grouped_top10)
    plt.title(f'Suma de AREA_HA por {col} (Top 10)')
    plt.xlabel(col)
    plt.ylabel('Suma de AREA_HA')
    
    # Personalizar las etiquetas del eje X:
    plt.xticks(rotation=90, fontsize=7, color='black')
    
    plt.tight_layout()
    plt.show()

################################################################################
unique_values2 = solicitud_df['TIPO_TERMI'].unique()
unique_values2 = [x for x in solicitud_df['TIPO_TERMI'].unique() if x is not None]


# 1. Crear el nuevo dataframe "solicitud_df02" filtrando filas donde "TIPO_TERMI" sea vacío
solicitud_df02 = solicitud_df[~solicitud_df['TIPO_TERMI'].isin(unique_values2)]

# 2. Convertir "FECHA_DE_S" a datetime y extraer el año
solicitud_df02['FECHA_DE_S'] = pd.to_datetime(solicitud_df02['FECHA_DE_S'], errors='coerce')
solicitud_df02['year'] = solicitud_df02['FECHA_DE_S'].dt.year

def plot_grouped_bar(data, group_var, top_n_per_year=None):
    """
    Esta función realiza las siguientes operaciones:
      1. Agrupa 'data' por 'year' y por la variable 'group_var', sumando la columna 'AREA_HA'.
         Si se especifica 'top_n_per_year', se filtra para conservar solo los n primeros registros
         por año (según la suma de AREA_HA).
      2. Genera un diagrama de barras (eje X: año, barras diferenciadas según 'group_var') que muestra
         la suma de AREA_HA.
      3. Imprime la tabla resultante de la agregación.
      4. Genera un gráfico de dispersión con los datos individuales (sin agregación) usando los mismos ejes:
         eje X = año y eje Y = AREA_HA, diferenciando por 'group_var'.
    """
    # Agrupar por 'year' y 'group_var' sumando AREA_HA
    grouped = data.groupby(['year', group_var])['AREA_HA'].sum().reset_index()
    # Asegurarse de que AREA_HA sea numérico
    grouped['AREA_HA'] = pd.to_numeric(grouped['AREA_HA'], errors='coerce')
    
    # Filtrar para conservar solo los top n por año, si se especifica
    if top_n_per_year is not None:
        def top_n(df):
            return df.sort_values('AREA_HA', ascending=False).head(top_n_per_year)
        grouped = grouped.groupby('year', group_keys=False).apply(top_n)
    
    # Pivotear la tabla para graficar el diagrama de barras
    pivot_df = grouped.pivot(index='year', columns=group_var, values='AREA_HA').fillna(0)
    
    # Diagrama de barras (suma de AREA_HA)
    pivot_df.plot(kind='bar', stacked=False, figsize=(10,6))
    plt.ylabel('Suma de AREA_HA')
    plt.title(f'Suma de AREA_HA por Año y {group_var}')
    plt.legend(title=group_var)
    plt.tight_layout()
    plt.show()
    
    # Imprimir la tabla resultante
    print(f"Tabla de Suma de AREA_HA por Año y {group_var}:")
    print(grouped.sort_values(['year', 'AREA_HA'], ascending=[True, False]))
    print("\n" + "="*80 + "\n")
    
    # Gráfico de dispersión (valores individuales de AREA_HA)
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data, x='year', y='AREA_HA', hue=group_var, palette="viridis")
    plt.ylabel('AREA_HA')
    plt.title(f'Gráfico de dispersión de AREA_HA por Año y {group_var}')
    plt.legend(title=group_var)
    plt.tight_layout()
    plt.show()

# Definir las columnas en las que se muestran todos los valores
cols_completas = ['PUBLICADO_', 'TENURE_STA', 'MODALIDAD', 'TITLE_TYPE','MINING_CLA', 'TITULO_EST', 'ETAPA', 'PAR']

# Generar diagrama de barras y tabla para cada una de las variables completas
for col in cols_completas:
    plot_grouped_bar(solicitud_df02, col)


# Lista de columnas a analizar
group_columns = ['DEPARTAMEN', 'MINERALES', 'MINERALES_']

for col in group_columns:
    # Agrupar por la columna y sumar AREA_HA
    grouped = solicitud_df02.groupby(col)['AREA_HA'].sum().reset_index()
    
    # Ordenar la tabla de mayor a menor según AREA_HA
    grouped = grouped.sort_values('AREA_HA', ascending=False)
    
    # Tomar solo los 10 primeros
    grouped_top10 = grouped.head(10)
    
    # Mostrar la tabla resultante
    print(f"Tabla de suma de AREA_HA por {col} (Top 10):")
    print(grouped_top10)
    print("\n" + "="*50 + "\n")
    
    # Crear el diagrama de barras para los 10 primeros
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x=col, y='AREA_HA', data=grouped_top10)
    plt.title(f'Suma de AREA_HA por {col} (Top 10)')
    plt.xlabel(col)
    plt.ylabel('Suma de AREA_HA')
    
    # Personalizar las etiquetas del eje X:
    plt.xticks(rotation=90, fontsize=7, color='black')
    
    plt.tight_layout()
    plt.show()
    

# 5. VISUALIZACIONES GEOGRÁFICAS BÁSICAS (OPCIONAL)
#    USANDO LA COLUMNA 'geometry' EN LOS GDF ORIGINALES
# ------------------------------------------------------------------
def quick_map(gdf, column=None, title="Mapa"):
    """
    Realiza un plot geoespacial simple:
      - Si se especifica 'column', colorea los polígonos/puntos
        de acuerdo con ese atributo.
    """
    plt.figure(figsize=(8,6))
    if column and column in gdf.columns:
        gdf.plot(column=column, legend=True, cmap='viridis')
    else:
        gdf.plot(edgecolor='black')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Mapas simples coloreando por área (si la columna existe y es relevante)
if 'AREA_HA' in solicitud_gdf.columns:
    quick_map(solicitud_gdf, column='AREA_HA', title="Mapa - Solicitud Vigente (Área)")

if 'AREA_HA' in subcontrato_gdf.columns:
    quick_map(subcontrato_gdf, column='AREA_HA', title="Mapa - Subcontrato (Área)")

if 'AREA_HA' in titulo_gdf.columns:
    quick_map(titulo_gdf, column='AREA_HA', title="Mapa - Título Vigente (Área)")
    
###########################################################
########### Imputar valores faltantes en los DataFrames #############

for df in [titulo_df, subcontrato_df, solicitud_df]:
    if 'TIPO_TERMI' in df.columns:
        # Para la columna TIPO_TERMI, llenar los NaN y vacíos con "Vigencia"
        df['TIPO_TERMI'] = df['TIPO_TERMI'].fillna("Vigencia")
        df['TIPO_TERMI'] = df['TIPO_TERMI'].replace("", "Vigencia")
    # Para todas las demás columnas, imputar los valores faltantes con "En_actualizacion"
    df.fillna("En_actualizacion", inplace=True)



