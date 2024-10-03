import pandas as pd
import numpy as np
import sys
import time
inicioTotal = time.time()
inicio = time.time()
#datos = pd.read_csv("Movie_ratings.csv",index_col=0)
#datos = pd.read_csv("ratings32M.csv")
#datos = pd.read_csv("ratings10M.dat",delimiter='::',engine="python",header=None); datos.columns = ['userId','movieId','rating','timestamp']
datos = pd.read_csv("ratings100K.csv")
#datos = pd.read_csv("ratings1M.dat",delimiter='::',engine="python",header=None); datos.columns = ['userId','movieId','rating','timestamp']
#datos = pd.read_csv("DatosCoseno.csv")
#datos = pd.read_csv("DatosSlopeOne.csv")
fin = time.time()
print(f"El tiempo que se tarda en la lectura de datos es: {fin - inicio}")

inicio = time.time()
datospivot = datos.pivot(index='movieId',columns='userId',values='rating')
fin = time.time()
print(f"El tiempo que se tarda en ordenar los datos es: {fin - inicio}")

#print(datos)
print(datospivot)
#datospivot.to_csv('10Mrating_actualizado.csv',index=True)


#Funciones Replanteadas
def manhattanColumna(tabla,col1,col2):
    columna1 = tabla.iloc[:,col1]
    columna2 = tabla.iloc[:,col2]
    print(columna1.values)
    print(columna2.values)
    distancia = np.sum(np.abs(columna1-columna2))
    print(f"La distancia Manhattan entre {tabla.columns[col1]} y {tabla.columns[col2]} es de: {distancia}")
    return distancia

def euclidesColumna(tabla,col1,col2):
    columna1 = tabla.iloc[:,col1]
    columna2 = tabla.iloc[:,col2]

    distancia = np.sqrt(np.sum(np.square(columna1-columna2)))
    print(f"La distancia Euclidiana entre {tabla.columns[col1]} y {tabla.columns[col2]} es de: {distancia}")
    return distancia

def pearsonColumna(tabla,col1,col2):
    columna1 = tabla.iloc[:,col1]
    columna2 = tabla.iloc[:,col2]

    distancia = columna1.corr(columna2)
    if pd.isna(distancia):
        print(f"La distancia Pearson entre {tabla.columns[col1]} y {tabla.columns[col2]} no es definida (NaN)")
    else:
        print(f"La distancia Pearson entre {tabla.columns[col1]} y {tabla.columns[col2]} es de: {distancia}")
    return distancia

def cosenoColumna(tabla,col1,col2):
    columna1 = tabla.iloc[:,col1]
    columna2 = tabla.iloc[:,col2]

    tablaAuxiliar = pd.concat([columna1,columna2],axis=1).dropna()
    #print(tablaAuxiliar)
    if len(tablaAuxiliar) == 0:
        raise ValueError("Las columnas no tienen datos validos, son NaN")
    #vector1 = tablaAuxiliar.iloc[:,0] sin el .values seguiriamos trabajando en una Serie(columna) de pandas
    #vector1 = tablaAuxiliar.iloc[:,0].values con el .values estariamos trabajando con un array de Numpy
    #Numpy es mejor en operaciones matematicas
    vector1 = tablaAuxiliar.iloc[:,0].values
    vector2 = tablaAuxiliar.iloc[:,1].values
    #print(vector1)
    #print(vector2)
    producto = np.dot(vector1,vector2)
    #print(producto)
    norma1 = np.linalg.norm(vector1)
    norma2 = np.linalg.norm(vector2)
    #print(norma1)
    #print(norma2)
    distancia = producto/(norma1 * norma2)
    print(f"La similitud de Coseno entre {tabla.columns[col1]} y {tabla.columns[col2]} es de: {distancia}")
    return distancia

#Funciones optimizadas
def manhattan(point1,point2):
    auxiliar = ~point1.isna() & ~point2.isna()
    if auxiliar.sum() == 0:
        return np.nan
    return np.sum(np.abs(point1 - point2))

def euclides(point1,point2):
    auxiliar = ~point1.isna() & ~point2.isna()
    if auxiliar.sum() == 0:
        return np.nan
    return np.sqrt(np.sum((point1 - point2)**2))

def pearson(point1,point2):
    auxiliar = ~point1.isna() & ~point2.isna()
    if auxiliar.sum() == 0:
        return np.nan
    return point1.corr(point2)

def coseno(point1,point2):
    auxiliar = ~point1.isna() & ~point2.isna()
    if auxiliar.sum() == 0:
        return np.nan
    tablaAuxiliar = pd.concat([point1,point2],axis=1).dropna()
    if len(tablaAuxiliar) == 0:
        return np.nan
    vector1 = tablaAuxiliar.iloc[:,0].values
    vector2 = tablaAuxiliar.iloc[:,1].values
    producto = np.dot(vector1,vector2)
    norma1 = np.linalg.norm(vector1)
    norma2 = np.linalg.norm(vector2)
    return producto/(norma1*norma2)


#Funciones Alternativas

def verificacion(point1,point2):
    mask = ~point1.isna() & ~point2.isna()
    
    # Extraemos los pares válidos
    valid_point1 = point1[mask]
    valid_point2 = point2[mask]
    
    # Imprimimos los pares que coinciden
    print("Pares válidos:")
    print(pd.DataFrame({"point1": valid_point1, "point2": valid_point2}))

#COSENO AJUSTADO
def coseno_ajustado(data,item1,item2):
    point1 = data.loc[item1]
    point2 = data.loc[item2]
    promedios = data.mean()
    
    vector1 = point1 - promedios
    vector2 = point2 - promedios

    auxiliar = ~vector1.isna() & ~vector2.isna()
    if auxiliar.sum() == 0:
        return np.nan
    
    vector1 = vector1[auxiliar]
    vector2 = vector2[auxiliar]

    producto = np.dot(vector1, vector2)
    norma1 = np.linalg.norm(vector1)
    norma2 = np.linalg.norm(vector2)

    return producto / (norma1 * norma2) if norma1 and norma2 else np.nan

def matrix_coseno_ajustado(data):
    items = data.index
    tabla_coseno = pd.DataFrame(index=items,columns=items)

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            item1 = items[i]
            item2 = items[j]

            tabla_coseno.loc[item1,item2] = coseno_ajustado(data,item1,item2)

    print(tabla_coseno)
    return tabla_coseno

def proyeccion_coseno_ajustado(data,usuario,item):
    if pd.notna(data.at[item,usuario]):
        print(f"El item {item} ya tiene un rating del usuario y es: {data.at[item,usuario]}")
        sys.exit()
    tabla_coseno = matrix_coseno_ajustado(data)
    rating_max = 5
    rating_min = 1
    auxiliar = data[usuario].notna()
    index_usuario = data.index[auxiliar]
    #print(auxiliar)
    rating_usuario = pd.DataFrame(index=index_usuario,columns=['R','NR','S'])
    rating_usuario['R'] = data[usuario][auxiliar]
    #Normalizacion
    rating_usuario['NR'] = (((2*(rating_usuario['R']-rating_min)-(rating_max - rating_min))/(rating_max - rating_min)))
    rating_usuario['S'] = rating_usuario.index.map(lambda a: tabla_coseno.loc[a, item] if pd.notna(tabla_coseno.loc[a, item]) else tabla_coseno.loc[item,a])
    print(rating_usuario)
    resultado = sum(rating_usuario['NR']*rating_usuario['S'])/sum(abs(rating_usuario['S']))
    print(f"Proyeccion de coseno ajustado: {resultado}")

def proyeccion_coseno_ajustado2(data,usuario,item):
    if pd.notna(data.at[item,usuario]):
        print(f"El item {item} ya tiene un rating del usuario y es: {data.at[item,usuario]}")
        sys.exit()
    #tabla_coseno = matrix_coseno_ajustado(data)
    rating_max = 5
    rating_min = 1
    auxiliar = data[usuario].notna()
    index_usuario = data.index[auxiliar]
    #print(auxiliar)
    rating_usuario = pd.DataFrame(index=index_usuario,columns=['R','NR','S'])
    rating_usuario['R'] = data[usuario][auxiliar]
    #Normalizacion
    rating_usuario['NR'] = (((2*(rating_usuario['R']-rating_min)-(rating_max - rating_min))/(rating_max - rating_min)))
    #rating_usuario['S'] = rating_usuario.index.map(lambda a: tabla_coseno.loc[a, item] if pd.notna(tabla_coseno.loc[a, item]) else tabla_coseno.loc[item,a])
    #rating_usuario['S'] = coseno_ajustado(data,index_usuario,item)
    rating_usuario['S'] = rating_usuario.index.map(lambda a: coseno_ajustado(data, a, item))
    #Elimina filas que su similitud 'S' sea NaN
    rating_usuario = rating_usuario.dropna(subset=['S'])
    print(rating_usuario)
    resultado = (sum(rating_usuario['NR']*rating_usuario['S']))/sum(abs(rating_usuario['S']))
    resultado_desnormalizado = 1/2*((resultado+1)*(rating_max - rating_min)) + rating_min
    print(f"Proyeccion de coseno ajustado: {resultado_desnormalizado}")

#SLOPE ONE
def slope_one(data,item1,item2):
    #Guardo en booleanos los usuarios que calificaron ambas peliculas
    usuarios_calificaron_ambos = data.loc[item1].notna() & data.loc[item2].notna()
    usuarios_comunes = data.columns[usuarios_calificaron_ambos]
    point1 = data.loc[item1, usuarios_calificaron_ambos]
    point2 = data.loc[item2, usuarios_calificaron_ambos]
    len_usuarios_comunes = len(usuarios_comunes)
    print("usuarios comunes")
    print(usuarios_comunes)
    print(point1)
    print(point2)
    resultado = sum((point1 - point2)/len_usuarios_comunes)
    print(resultado)
    return resultado, len_usuarios_comunes

def proyeccion_slope_one(data, usuario, item):
    print("---------------")
    calificadas_usuario_index = data.index[data[usuario].notna()]
    tabla_proyeccion = pd.DataFrame(index=calificadas_usuario_index, columns=['rating', 'deviation', 'both'])
    
    tabla_proyeccion['rating'] = data[usuario][calificadas_usuario_index]
    
    # Aplicar slope_one y obtener deviation y both
    slope_results = tabla_proyeccion.index.map(lambda i: slope_one(data, item, i))
    
    # Convertir los resultados en dos columnas
    tabla_proyeccion['deviation'] = [res[0] for res in slope_results]
    tabla_proyeccion['both'] = [res[1] for res in slope_results]
    
    print(tabla_proyeccion)
    
    # Calcular el resultado
    resultado = (sum((tabla_proyeccion['rating'] + tabla_proyeccion['deviation'])*tabla_proyeccion['both']))/(sum(tabla_proyeccion['both']))
    print(resultado)


def pearsonArreglado(point1,point2):
    try:
        correlation = point1.corr(point2)
    except (ZeroDivisionError, ValueError, TypeError):
        return np.nan
    return correlation

def knnMejorado(data,posicion,k,distancia_funcion):
    distancias = []
    for n in data.columns:
        if n == posicion:
            continue
        distancia = distancia_funcion(data[posicion],data[n])
        if pd.isna(distancia):
            continue
        distancias.append((n,distancia))
    if distancia_funcion == pearson or distancia_funcion == coseno:
        distancias.sort(key=lambda x: x[1],reverse=True)
    else:
        distancias.sort(key=lambda x: x[1])
    distancias_cercanas = distancias[:k]
    print("No\tNombre\t\tDistancia")
    for n in range(len(distancias_cercanas)):
        print(f"{n+1}\t{distancias_cercanas[n][0]:<15}\t{distancias_cercanas[n][1]:.13f}")
    return distancias_cercanas

def recomendacion(data,pelicula, lista):
    total = sum(lista[1] for lista in lista)
    dataaux = pd.DataFrame(columns=['usuario','rating','influencia'])
    for n in range(len(lista)):
        dataaux.at[n, 'usuario'] = lista[n][0]
        dataaux.at[n, 'rating'] = data.at[pelicula,lista[n][0]]
        #print(data.at[pelicula,lista[n][0]])
        dataaux.at[n, 'influencia'] = lista[n][1]/total
        #print("-----------------")
        #print(data.at[pelicula, lista[n][0]]) #EL ERROR ES PORQUE ESTOY PONIENDO USUARIO CON USUARIO y no CON PELICULA
    
    resultado = (dataaux['rating'] * dataaux['influencia']).sum(skipna=True)
    
    print(dataaux)
    print("-----------------")
    print(f"Projected rating: {resultado}")
    print("-----------------")

def recomendacionMejorado(data,pelicula,umbral,usuario,lista):
    #total = sum(lista[1] for lista in lista)
    total = 0
    dataaux = pd.DataFrame(columns=['usuario','rating','influencia'])
    for n in range(len(lista)):
        if pd.isna(data.at[pelicula,lista[n][0]]):
            continue
        if data.at[pelicula,lista[n][0]] < umbral:
            continue
        if lista[n][1] <= 0:
            continue 
        dataaux.at[n, 'usuario'] = lista[n][0]
        dataaux.at[n, 'rating'] = data.at[pelicula,lista[n][0]]
        total = total + lista[n][1]
        #print(data.at[pelicula,lista[n][0]])
        #dataaux.at[n, 'influencia'] = lista[n][1]/total
    
    for n in range(len(lista)):
        if pd.isna(data.at[pelicula,lista[n][0]]):
            continue
        if data.at[pelicula,lista[n][0]] < umbral:
            continue
        if lista[n][1] <= 0:
            continue 
        dataaux.at[n,'influencia'] = lista[n][1]/total


    resultado = (dataaux['rating'] * dataaux['influencia']).sum(skipna=True)
    
    print(dataaux)
    print("-----------------")
    if pd.notna(data.at[pelicula,usuario]):
        print(f"El usuario ya califico esta pelicula con: {data.at[pelicula,usuario]}")
    print(f"Projected rating: {resultado}")
    print(f"Total: {total}")
    print("-----------------")
#MAIN

inicio = time.time()
#print(f"Este es el dato que buscabas {datospivot.at[10,6027]}")
#print(coseno_ajustado(datospivot,52245,95858))
#proyeccion_coseno_ajustado2(datospivot,6,10)
#slope_one(datospivot,193609,26052)
#proyeccion_slope_one(datospivot,132,170875)

#-------------------------IDPelicula,Umbral,IDUsuario--------IDUsuario,K,FuncionDistancia
recomendacionMejorado(datospivot,170875,3,132,knnMejorado(datospivot,132,20,euclides))

fin = time.time()
print(f"El tiempo que se tarda en knn es: {fin - inicio}")
#print(coseno(datospivot[18388],datospivot[69000]))
finTotal = time.time()
print(f"El tiempo total del proceso fue de: {finTotal - inicioTotal}")

