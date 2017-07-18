# coding=utf-8
# ==========================================================
# Ampliación de Inteligencia Artificial. Tercer curso.
# Grado en Ingeniería Informática - Tecnologías Informáticas
# Curso 2016-17
# Trabajo práctico
# ===========================================================

# --------------------------------------------------------------------------
# Autor del trabajo:
#
# APELLIDOS: Tarrio Gete
# NOMBRE: Santiago
#
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt

# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo que
# debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite (e incluso se
# recomienda), pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de
# terceros, OBTENIDO A TRAVÉS DE LA RED o cualquier otro medio, se considerará
# plagio.

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de CERO EN LA ASIGNATURA para TODOS los
# alumnos involucrados. Por tanto a estos alumnos NO se les conservará, para
# futuras convocatorias, ninguna nota que hubiesen obtenido hasta el
# momento. SIN PERJUICIO DE OTRAS MEDIDAS DE CARÁCTER DISCIPLINARIO QUE SE
# PUDIERAN TOMAR.
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES Y MÉTODOS
# QUE SE PIDEN

# NOTA: En este trabajo no se permite usar scikit-learn

# ====================================================
# PARTE I: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# ====================================================

# En esta primera parte se pide implementar en Python los siguientes
# clasificadores BINARIOS, todos ellos vistos en el tema 5.

# - Perceptron umbral
# - Regresión logística minimizando el error cuadrático:
#      * Versión batch
#      * Versión estocástica (regla delta)
# - Regresión logística maximizando la verosimilitud:
#      * Versión batch
#      * Versión estocástica


# Imports
import random
import numpy as np

# --------------------------------------------
# I.1. Generando conjuntos de datos aleatorios
# --------------------------------------------

# Previamente a la implementación de los clasificadores, conviene tener
# funciones que generen aleatoriamente conjuntos de datos fictícios.
# En concreto, se pide implementar estas dos funciones:

# * Función genera_conjunto_de_datos_l_s(rango,dim,n_datos):

#   Debe devolver dos listas X e Y, generadas aleatoriamente. La lista X debe
#   tener un número total n_datos de elelemntos, siendo cada uno de ellos una
#   lista (un ejemplo) de dim componentes, con valores entre -rango y rango. El
#   conjunto Y debe tener la clasificación binaria (1 o 0) de cada ejemplo del
#   conjunto X, en el mismo orden. El conjunto de datos debe ser linealmente
#   separable.

#   SUGERENCIA: generar en primer lugar un hiperplano aleatorio (mediante sus
#   coeficientes, elegidos aleatoriamente entre -rango y rango). Luego generar
#   aleatoriamente cada ejemplo de igual manera y clasificarlo como 1 o 0
#   dependiendo del lado del hiperplano en el que se situe. Eso asegura que el
#   conjunto de datos es linealmente separable.

def genera_conjunto_de_datos_l_s(rango,dim,n_datos):
    hyperplane = np.random.uniform(-rango,rango,dim)
    X = []
    Y = []
    for _ in  range(n_datos):
        nuevo_ejemplo = np.random.uniform(-rango,rango,dim)
        X.append(nuevo_ejemplo)
        clase_n_e = 1 if sum(hyperplane*nuevo_ejemplo) > 0 else 0
        Y.append(clase_n_e)
    return (X,Y)



# * Función genera_conjunto_de_datos_n_l_s(rango,dim,size,prop_n_l_s=0.1):

#   Como la anterior, pero el conjunto de datos debe ser no linealmente
#   separable. Para ello generar el conjunto de datos con la función anterior
#   y cambiar de clase a una proporción pequeña del total de ejemplos (por
#   ejemplo el 10%). La proporción se da con prop_n_l_s.

def genera_conjunto_de_datos_n_l_s(rango,dim,size,prop_n_l_s=0.1):
    (X,Y)=genera_conjunto_de_datos_l_s(rango,dim,size)
    for i in range(int(prop_n_l_s*size)):
        Y[i] = 0 if Y[i]==1 else 1
    return (X,Y)






# -----------------------------------
# I.2. Clases y métodos a implementar
# -----------------------------------

# En esta sección se pide implementar cada uno de los clasificadores lineales
# mencionados al principio. Cada uno de estos clasificadores se implementa a
# través de una clase python, que ha de tener la siguiente estructura general:

# class NOMBRE_DEL_CLASIFICADOR():

#     def __init__(self,clases,normalizacion=False):

#          .....

#     def entrena(self,entr,clas_entr,n_epochs,rate=0.1,
#                 pesos_iniciales=None,
#                 rate_decay=False):

#         ......

#     def clasifica_prob(self,ej):


#         ......

#     def clasifica(self,ej):


#         ......


# Explicamos a continuación cada uno de estos elementos:

# * NOMBRE_DEL_CLASIFICADOR:
# --------------------------


#  Este es el nombre de la clase que implementa el clasificador.
#  Obligatoriamente se han de usar cada uno de los siguientes
#  nombres:

#  - Perceptrón umbral:
#                       Clasificador_Perceptron
#  - Regresión logística, minimizando L2, batch:
#                       Clasificador_RL_L2_Batch
#  - Regresión logística, minimizando L2, estocástico:
#                       Clasificador_RL_L2_St
#  - Regresión logística, maximizando verosimilitud, batch:
#                       Clasificador_RL_ML_Batch
#  - Regresión logística, maximizando verosimilitud, estocástico:
#                       Clasificador_RL_ML_St



# * Constructor de la clase:
# --------------------------

#  El constructor debe tener los siguientes argumentos de entrada:

#  - Una lista clases con los nombres de las clases del problema de
#    clasificación, tal y como aparecen en el conjunto de datos.
#    Por ejemplo, en el caso del problema de las votaciones,
#    esta lista sería ["republicano","democrata"]

#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en cada componente i-esima (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta misma transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.  NOTA: se permite
#    usar la biblioteca numpy para calcular la media, la desviación típica, y
#    en general para cualquier cálculo matemático.



# * Método entrena:
# -----------------

#  Este método es el que realiza el entrenamiento del clasificador.
#  Debe calcular un conjunto de pesos, mediante el correspondiente
#  algoritmo de entrenamiento. Describimos a continuación los parámetros de
#  entrada:

#  - entr y clas_entr, son los datos del conjunto de entrenamiento y su
#    clasificación, respectivamente. El primero es una lista con los ejemplos,
#    y el segundo una lista con las clasificaciones de esos ejemplos, en el
#    mismo orden.

#  - n_epochs: número de veces que se itera sobre todo el conjunto de
#    entrenamiento.

#  - rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate marca una cota
#    mínima de la tasa de aprendizaje, como se explica a continuación.

#  - rate_decay, indica si la tasa de aprendizaje debe disminuir a medida que
#    se van realizando actualizaciones de los pases. En concreto, si
#    rate_decay es True, la tasa de aprendizaje que se usa en cada
#    actualización se debe de calcular con la siguiente fórmula:
#       rate_n= rate_0 + (2/n**(1.5))
#    donde n es el número de actualizaciones de pesos realizadas hasta el
#    momento, y rate_0 es la cantidad introducida en el parámetro rate
#    anterior.
#
#  - pesos_iniciales: si es None, se indica que los pesos deben iniciarse
#    aleatoriamente (por ejemplo, valores aleatorios entre -1 y 1). Si no es
#    None, entonces se debe proporcionar la lista de pesos iniciales. Esto
#    puede ser útil para continuar el aprendizaje a partir de un aprendizaje
#    anterior, si por ejemplo se dispone de nuevos datos.

#  NOTA: En las versiones estocásticas, y en el perceptrón umbral, en cada
#  epoch recorrer todos los ejemplos del conjunto de entrenamiento en un orden
#  aleatorio distinto cada vez.


# * Método clasifica_prob:
# ------------------------

#  El método que devuelve la probabilidad de pertenecer a la clase (la que se
#  ha tomado como clase 1), calculada para un nuevo ejemplo. Este método no es
#  necesario incluirlo para el perceptrón umbral.



# * Método clasifica:
# -------------------

#  El método que devuelve la clase que se predice para un nuevo ejemplo. La
#  clase debe ser una de las clases del problema (por ejemplo, "republicano" o
#  "democrata" en el problema de los votos).


# Si el clasificador aún no ha sido entrenado, tanto "clasifica" como
# "clasifica_prob" deben devolver una excepción del siguiente tipo:

class ClasificadorNoEntrenado(Exception): pass

#  NOTA: Se aconseja probar el funcionamiento de los clasificadores con
#  conjuntos de datos generados por las funciones del apartado anterior.

# Ejemplo de uso:

# ------------------------------------------------------------

# Generamos un conjunto de datos linealmente separables,
# In [1]: X1,Y1,w1=genera_conjunto_de_datos_l_s(4,8,400)

# Lo partimos en dos trozos:
# In [2]: X1e,Y1e=X1[:300],Y1[:300]

# In [3]: X1t,Y1t=X1[300:],Y1[300:]

# Creamos el clasificador (perceptrón umbral en este caso):
# In [4]: clas_pb1=Clasificador_Perceptron([0,1])

# Lo entrenamos con elprimero de los conjuntos de datos:
# In [5]: clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)

# Clasificamos un ejemplo del otro conjunto, y lo comparamos con su clase real:
# In [6]: clas_pb1.clasifica(X1t[0]),Y1t[0]
# Out[6]: (1, 1)

# Comprobamos el porcentaje de aciertos sobre todos los ejemplos de X2t
# In [7]:
# Out[7]: 1.0

# Repetimos el experimento, pero ahora con un conjunto de datos que no es
# linealmente separable:
# In [8]: X2,Y2,w2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)

# In [8]: X2e,Y2e=X2[:300],Y2[:300]

# In [9]: X2t,Y2t=X2[300:],Y2[300:]

# In [10]: clas_pb2=Clasificador_Perceptron([0,1])

# In [11]: clas_pb2.entrena(X2e,Y2e,100,rate_decay=True,rate=0.001)

# In [12]: clas_pb2.clasifica(X2t[0]),Y2t[0]
# Out[12]: (1, 0)

# In [13]: sum(clas_pb2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t)
# Out[13]: 0.82
# ----------------------------------------------------------------

#  - Perceptrón umbral:
#                       Clasificador_Perceptron
#  - Regresión logística, minimizando L2, batch:
#                       Clasificador_RL_L2_Batch
#  - Regresión logística, minimizando L2, estocástico:
#                       Clasificador_RL_L2_St
#  - Regresión logística, maximizando verosimilitud, batch:
#                       Clasificador_RL_ML_Batch
#  - Regresión logística, maximizando verosimilitud, estocástico:
#                       Clasificador_RL_ML_St


## PERCEPTRON UMBRAL

class Clasificador_Perceptron():

    def __init__(self,clases,normalizacion=False):
        self.clases = clases
        self.normalizacion = normalizacion
        self.errores = []
        self.label_encoder = label_fit(self.clases)

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1, pesos_iniciales=None, rate_decay=False):

        #Esta parte sirve para preparar los datos antes de que puedan ser aplicados en el algoritmo
        if type(entr[0]) is list:
            #Convertimos las listas de datos a arrays Numpy
            entr = [np.array(x) for x in entr]
        if self.normalizacion:
            #Aplicamos la normalización si corresponde
            self.medias_i, self.std_i = [np.mean([e[i] for e in entr]) for i in range(entr[0].size)], [np.std([e[i] for e in entr]) for i in range(entr[0].size)]
            entr = normalizacion(entr,self.medias_i,self.std_i)
        if type(clas_entr) is list:
            #Convertimos el vector de variables respuesta a un array numérico que pueda usar el algoritmo
            clas_entr = label_transform(clas_entr,self.label_encoder)
        if not hasattr(self,'pesos'):
            #Evita inicializar los pesos de forma aleatoria cada vez que se inicia una nueva tanda de entrenamiento
            self.pesos = pesos_iniciales if pesos_iniciales != None else np.random.uniform(-1,1,entr[0].size)


        rate_0 = rate

        #Esta lista servirá, en todos los algoritmos estocásticos, para poder variar el orden en el que
        # entrenamos cada epoch
        random_index_list = list(range(len(entr)))

        for n in range(n_epochs):

            # Randomizamos el conjunto de entrenamiento en cada iteración
            random.shuffle(random_index_list)
            for i in random_index_list:
                o_i = self.umbral(sum(self.pesos*entr[i]))
                for j in range(entr[i].size):
                    self.pesos[j] += rate*entr[i][j]*(clas_entr[i] - o_i)
            self.errores.append(1.0-(sum(self.clasifica(x) == y for x,y in zip(entr,clas_entr))/len(clas_entr)))
            if(rate_decay and n!=0):
                rate = rate_0 + (2/n**(1.5))

    def clasifica(self,ej):
        if not hasattr(self,'pesos'):
            # Evita que se pueda clasificar con un modelo no entrenado
            raise ClasificadorNoEntrenado("Clasificador no entrenado")
        ej = np.array(ej)
        if self.normalizacion:
            for i in range(ej.size):
                ej[i]=(ej[i] - medias_i[i]) / std_i[i]
        return self.clases[self.umbral(sum(self.pesos*ej))]

    def umbral(self,n):
        return 1 if n>=0 else 0


## REGRESION LOGISTICA MINIMIZANDO L2 BATCH

class Clasificador_RL_L2_Batch():

    def __init__(self,clases,normalizacion=False):
        self.clases = clases
        self.normalizacion = normalizacion
        self.errores = []
        self.label_encoder = label_fit(self.clases)

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1, pesos_iniciales=None, rate_decay=False):
        if type(entr[0]) is list:
            entr = [np.array(x) for x in entr]
        if self.normalizacion:
            self.medias_i, self.std_i = [np.mean([e[i] for e in entr]) for i in range(entr[0].size)], [np.std([e[i] for e in entr]) for i in range(entr[0].size)]
            entr = normalizacion(entr,self.medias_i,self.std_i)
        if type(clas_entr) is list:
            clas_entr = label_transform(clas_entr,self.label_encoder)
        if not hasattr(self,'pesos'):
            self.pesos = pesos_iniciales if pesos_iniciales != None else np.random.uniform(-1,1,entr[0].size)

        rate_0 = rate
        for e in range(n_epochs):
            self.pesos += rate*np.array([ sum(entr[j][i] * (clas_entr[j] - self.sigmoid(sum(self.pesos*entr[j][i]))) for j in range(len(entr))) for i in range(self.pesos.size) ])
            self.errores.append(1.0-(sum(self.clasifica_prob(x) == y for x,y in zip(entr,clas_entr))/len(clas_entr)))
            if(rate_decay and e!=0):
                rate = rate_0 + (2/e**(1.5))


    def clasifica_prob(self,ej):
        if not hasattr(self,'pesos'):
            raise ClasificadorNoEntrenado("Clasificador no entrenado")

        ej = np.array(ej)
        if self.normalizacion:
            for i in range(ej.size):
                ej[i]=(ej[i] - medias_i[i]) / std_i[i]
        return self.sigmoid(sum(self.pesos*ej))

    def clasifica(self,ej):
        return self.clases[1 if self.clasifica_prob(ej)>=0.5 else 0]

    def sigmoid(self,z):
        return 1.0/(1.0 + np.exp(-z))



# ## REGRESION LOGISTICA MINIMIZANDO L2 ESTOCASTICO

class Clasificador_RL_L2_St():

    def __init__(self,clases,normalizacion=False):
        self.clases = clases
        self.normalizacion = normalizacion
        self.errores = []
        self.label_encoder = label_fit(self.clases)

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1, pesos_iniciales=None, rate_decay=False):
        if type(entr[0]) is list:
            entr = [np.array(x) for x in entr]
        if self.normalizacion:
            self.medias_i, self.std_i = [np.mean([e[i] for e in entr]) for i in range(entr[0].size)], [np.std([e[i] for e in entr]) for i in range(entr[0].size)]
            entr = normalizacion(entr,self.medias_i,self.std_i)
        if type(clas_entr) is list:
            clas_entr = label_transform(clas_entr,self.label_encoder)
        if not hasattr(self,'pesos'):
            self.pesos = pesos_iniciales if pesos_iniciales != None else np.random.uniform(-1,1,entr[0].size)

        rate_0 = rate
        random_index_list = list(range(len(entr)))
        for n in range(n_epochs):
            random.shuffle(random_index_list)
            for i in random_index_list:
                o_i = self.sigmoid(sum(self.pesos*entr[i]))
                for j in range(entr[i].size):
                    self.pesos[j] += rate*entr[i][j]*(clas_entr[i] - o_i)*o_i*(1-o_i)
            self.errores.append(1.0-(sum(self.clasifica(x) == y for x,y in zip(entr,clas_entr))/len(clas_entr)))
            if(rate_decay and n!=0):
                rate = rate_0 + (2/n**(1.5))

    def clasifica_prob(self,ej):
        if not hasattr(self,'pesos'):
            raise ClasificadorNoEntrenado("Clasificador no entrenado")
        ej = np.array(ej)
        if self.normalizacion:
            for i in range(ej.size):
                ej[i]=(ej[i] - medias_i[i]) / std_i[i]
        return self.sigmoid(sum(self.pesos*ej))

    def clasifica(self,ej):
        return self.clases[1 if self.clasifica_prob(ej)>=0.5 else 0]

    def sigmoid(self,z):
        return 1.0/(1.0 + np.exp(-z))



# ## REGRESION LOGISTICA MAXIMIZANDO VEROSIMILITUD BATCH

class Clasificador_RL_ML_Batch():

    def __init__(self,clases,normalizacion=False):
        self.clases = clases
        self.normalizacion = normalizacion
        self.errores = []
        self.label_encoder = label_fit(self.clases)

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1, pesos_iniciales=None, rate_decay=False):
        if type(entr[0]) is list:
            entr = [np.array(x) for x in entr]
        if self.normalizacion:
            self.medias_i, self.std_i = [np.mean([e[i] for e in entr]) for i in range(entr[0].size)], [np.std([e[i] for e in entr]) for i in range(entr[0].size)]
            entr = normalizacion(entr,self.medias_i,self.std_i)
        if type(clas_entr) is list:
            clas_entr = label_transform(clas_entr,self.label_encoder)
        if not hasattr(self,'pesos'):
            self.pesos = pesos_iniciales if pesos_iniciales != None else np.random.uniform(-1,1,entr[0].size)
        rate_0 = rate
        for e in range(n_epochs):
            self.pesos += rate*np.array([ sum( (clas_entr[j] - self.sigmoid(sum(self.pesos*entr[j][i]))) * entr[j][i] for j in range(len(entr))) for i in range(self.pesos.size) ])
            self.errores.append(1.0-(sum(self.clasifica_prob(x) == y for x,y in zip(entr,clas_entr))/len(clas_entr)))
            if(rate_decay and e!=0):
                rate = rate_0 + (2/e**(1.5))

    def clasifica_prob(self,ej):
        if not hasattr(self,'pesos'):
            raise ClasificadorNoEntrenado("Clasificador no entrenado")
        ej = np.array(ej)
        if self.normalizacion:
            for i in range(ej.size):
                ej[i]=(ej[i] - medias_i[i]) / std_i[i]
        return self.sigmoid(sum(self.pesos*ej))

    def clasifica(self,ej):
        return self.clases[1 if self.clasifica_prob(ej)>=0.5 else 0]

    def sigmoid(self,z):
        return 1.0/(1.0 + np.exp(-z))




# ## REGRESION LOGISTICA MAXIMIZANDO VEROSIMILITUD ESTOCASTICO

class Clasificador_RL_ML_St():

    def __init__(self,clases,normalizacion=False):
        self.clases = clases
        self.normalizacion = normalizacion
        self.errores = []
        self.label_encoder = label_fit(self.clases)

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1, pesos_iniciales=None, rate_decay=False):
        if type(entr[0]) is list:
            entr = [np.array(x) for x in entr]
        if self.normalizacion:
            self.medias_i, self.std_i = [np.mean([e[i] for e in entr]) for i in range(entr[0].size)], [np.std([e[i] for e in entr]) for i in range(entr[0].size)]
            entr = normalizacion(entr,self.medias_i,self.std_i)
        if type(clas_entr) is list:
            clas_entr = label_transform(clas_entr,self.label_encoder)
        if not hasattr(self,'pesos'):
            self.pesos = pesos_iniciales if pesos_iniciales != None else np.random.uniform(-1,1,entr[0].size)

        rate_0 = rate
        random_index_list = list(range(len(entr)))
        for n in range(n_epochs):
            random.shuffle(random_index_list)
            for i in random_index_list:
                o_i = self.sigmoid(sum(self.pesos*entr[i]))
                for j in range(entr[i].size):
                    self.pesos[j] += rate*entr[i][j]*(clas_entr[i] - o_i)
            self.errores.append(1.0-(sum(self.clasifica(x) == y for x,y in zip(entr,clas_entr))/len(clas_entr)))
            if(rate_decay and n!=0):
                rate = rate_0 + (2/n**(1.5))

    def clasifica_prob(self,ej):
        if not hasattr(self,'pesos'):
            raise ClasificadorNoEntrenado("Clasificador no entrenado")
        ej = np.array(ej)
        if self.normalizacion:
            for i in range(ej.size):
                ej[i]=(ej[i] - medias_i[i]) / std_i[i]
        return self.sigmoid(sum(self.pesos*ej))

    def clasifica(self,ej):
        return self.clases[1 if self.clasifica_prob(ej)>=0.5 else 0]

    def sigmoid(self,z):
        return 1.0/(1.0 + np.exp(-z))




## Clases de preprocesado
# Como nuestra implementación se sirve de Numpy Arrays y no de listas, necesitamos convertir todos los datos en
# vectores numéricos antes de que puedan ser utilizados para entrenar los modelos. Para ello implementaremos
# funciones que sirvan a tal efecto.

def label_fit(clases):
    labels = {}
    n=0
    for clase in clases:
        labels[clase] = n
        n+=1
    return labels

def label_transform(y,encoding):
    return np.array([encoding[c] for c in y])

#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en cada componente i-esima (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta misma transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.  NOTA: se permite
#    usar la biblioteca numpy para calcular la media, la desviación típica, y
#    en general para cualquier cálculo matemático.

def normalizacion(X,medias_i,std_i):
# X es una lista de arrays, cada uno un ejemplo del conjunto
    for x in X:
        for i in range(x.size):
            x[i]=(x[i] - medias_i[i]) / std_i[i]
    return X


















# --------------------------
# I.3. Curvas de aprendizaje
# --------------------------

# Se pide mostrar mediante gráficas la evolución del aprendizaje de los
# distintos algoritmos. En concreto, para cada clasificador usado con un
# conjunto de datos generado aleatoriamente con las funciones anteriores, las
# dos siguientes gráficas:

# - Una gráfica que indique cómo evoluciona el porcentaje de errores que
#   comete el clasificador sobre el conjunto de entrenamiento, en cada epoch.
# - Otra gráfica que indique cómo evoluciona el error cuadrático o la log
#   verosimilitud del clasificador (dependiendo de lo que se esté optimizando
#   en cada proceso de entrenamiento), en cada epoch.

# Para realizar gráficas, se recomiendo usar la biblioteca matplotlib de
# python:

#import matplotlib.pyplot as plt


# Lo que sigue es un ejemplo de uso, para realizar una gráfica sencilla a
# partir de una lista "errores", que por ejemplo podría contener los sucesivos
# porcentajes de error que comete el clasificador, en los sucesivos epochs:


# plt.plot(range(1,len(errores)+1),errores,marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Porcentaje de errores')
# plt.show()

# Basta con incluir un código similar a este en el fichero python, para que en
# la terminal de Ipython se genere la correspondiente gráfica.

# Se pide generar una serie de gráficas que permitan explicar el
# comportamiento de los algoritmos, con las distintas opciones, y con
# conjuntos separables y no separables. Comentar la interpretación de las
# distintas gráficas obtenidas.

# NOTA: Para poder realizar las gráficas, debemos modificar los
# algoritmos de entrenamiento para que ademas de realizar el cálculo de los
# pesos, también calcule las listas con los sucesivos valores (de errores, de
# verosimilitud,etc.) que vamos obteniendo en cada epoch. Esta funcionalidad
# extra puede enlentecer algo el proceso de entrenamiento y si fuera necesario
# puede quitarse una vez se realize este apartado.

# Generamos los conjuntos de datos aleatorios tanto linealmente separables como no

(X,y) = genera_conjunto_de_datos_l_s(3,2,400)
Xe,ye = X[:300],y[:300]
Xt,yt = X[300:],y[300:]
(X1,y1) = genera_conjunto_de_datos_n_l_s(3,2,400)
X1e,y1e = X1[:300],y1[:300]
X1t,y1t = X1[300:],y1[300:]

## PERCEPTRON
perceptron1 = Clasificador_Perceptron([0,1])
perceptron1.entrena(Xe,ye,500)
plt.plot(range(1,len(perceptron1.errores)+1),perceptron1.errores,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Porcentaje de errores')
plt.show()

#En este caso vemos que el perceptron encuentra rápidamente el hiperplano que separa las dos clases, convergiendo a
# un vector de pesos que no comete errores sobre el conjunto de entrenamiento

perceptron2 = Clasificador_Perceptron([0,1])
perceptron2.entrena(X1e,y1e,1000,rate_decay=True,rate=0.0001)
plt.plot(range(1,len(perceptron2.errores)+1),perceptron2.errores,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Porcentaje de errores')
plt.show()

# En este caso vemos que los errores son completamente aleatorios y suben y bajan sin ningún sentido.
# Esto es porque el perceptrón no es capaz de hallar el hiperplano (porque no existe), y no está preparado
# para enfrentarse a un conjunto de datos que no sea linealmente separable

## REGRESION LOGISTICA BATCH, minimizando L2
clas_lineal_batch = Clasificador_RL_L2_Batch([0,1])
clas_lineal_batch.entrena(Xe,ye,300)
plt.plot(range(1,len(clas_lineal_batch.errores)+1),clas_lineal_batch.errores,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Porcentaje de errores')
plt.show()

# Este clasificador también converge a una distribución de pesos con muy poco error, pero no lo hace tan bien como
# el perceptrón. En general las versiones Batch durante todo este trabajo han ofrecido un menor rendimiento
# que el perceptrón o las versiones estocásticas.

rl2 = Clasificador_RL_L2_Batch([0,1])
rl2.entrena(X1e,y1e,100,rate_decay=True,rate=0.0001)
plt.plot(range(1,len(rl2.errores)+1),rl2.errores,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Porcentaje de errores')
plt.show()

# El clasificador lo hace un poco peor con el conjunto NLS, pero ofrece alrededor de un 60% de rendimiento sobre el
# conjunto de entrenamiento.
rl2.entrena(X1e,y1e,100,rate_decay=True,rate=0.0001)
#Con otra tanda de entrenamiento vemos que el error se reduce a cerca de un 30%
# Tras 1000 iteraciones de entrenamiento, el error final ronda el 20%

## REGRESIÓN LOGÍSTICA ESTOCÁSTICA, MINIMIZANDO L2

clas_lineal_estocastico = cl.Clasificador_RL_L2_St([0,1])
clas_lineal_estocastico.entrena(Xe,ye,300)
plt.plot(range(1,len(clas_lineal_estocastico.errores)+1),clas_lineal_estocastico.errores,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Porcentaje de errores')
plt.show()

## Aquí vemos que desde el principio el clasificador comete muy poco error y lo va reduciendo hasta llegar a 0

rb2 = cl.Clasificador_RL_L2_St([0,1])
rb2.entrena(X1e,y1e,100,rate_decay=True,rate=0.0001)
plt.plot(range(1,len(rb2.errores)+1),rb2.errores,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Porcentaje de errores')
plt.show()

## En este caso, el error comienza bastante alto pero se va reduciendo a medida que avanzamos en las iteraciones
# hasta ser prácticamente nulo

## REGRESION LOGISTICA BATCH, MAXIMIZANDO VEROSIMILITUD

rlvb = cl.Clasificador_RL_ML_Batch([0,1])
rlvb.entrena(Xe,ye,100,rate_decay=True,rate=0.0001)
plt.plot(range(1,len(rlvb.errores)+1),rlvb.errores,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Porcentaje de errores')
plt.show()

## En este caso nos encontramos un resultado no muy bueno. Reduce rápidamente el error pero luego vuelve a subir,
# y tras esto, la minimización del error es mas lenta y no llega tan lejos, estancándose en 0.5

rlvb2 = cl.Clasificador_RL_ML_Batch([0,1])
rlvb2.entrena(X1e,y1e,400,rate_decay=True,rate=0.0001)
plt.plot(range(1,len(rlvb2.errores)+1),rlvb2.errores,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Porcentaje de errores')
plt.show()

## En la versión no linealmente separable da un resultado aún mas pobre, manteniendo un 0.7 de error

## REGRESION LOGISTICA ESTOCASTICA, MAXIMIZANDO VEROSIMILITUD

rlve = cl.Clasificador_RL_ML_St([0,1])
rlve.entrena(Xe,ye,100,rate_decay=True,rate=0.0001)
plt.plot(range(1,len(rlve.errores)+1),rlve.errores,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Porcentaje de errores')
plt.show()

# Como siempre, la versión estocástica es tremendamente eficiente y reduce el error completamente en el caso
# linealmente separable.

rlve2 = cl.Clasificador_RL_ML_St([0,1])
rlve2.entrena(X1e,y1e,300,rate_decay=True,rate=0.0001)
plt.plot(range(1,len(rlve2.errores)+1),rlve2.errores,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Porcentaje de errores')
plt.show()

# En el caso no linealmente separable nos encontramos un comportamiento mas errático, pero
# con un error bastante bajo, aunque se dispare en algunos picos. A veces estos picos son provocados
# al iniciar una nueva tanda de entrenamiento, creemos que puede ser porque se vuelve a utilizar el learning
# rate inicial.








# ==================================
# PARTE II: CLASIFICACIÓN MULTICLASE
# ==================================

# Se pide implementar algoritmos de regresión logística para problemas de
# clasificación en los que hay más de dos clases. Para ello, usar las dos
# siguientes aproximaciones:

# ------------------------------------------------
# II.1 Técnica "One vs Rest" (Uno frente al Resto)
# ------------------------------------------------

#  Esta técnica construye un clasificador multiclase a partir de
#  clasificadores binarios que devuelven probabilidades (como es el caso de la
#  regresión logística). Para cada posible valor de clasificación, se
#  entrena un clasificador que estime cómo de probable es pertemecer a esa
#  clase, frente al resto. Este conjunto de clasificadores binarios se usa
#  para dar la clasificación de un ejemplo nuevo, sin más que devolver la
#  clase para la que su correspondiente clasificador binario da una mayor
#  probabilidad.

#  En concreto, se pide implementar una clase python Clasificador_RL_OvR con
#  la siguiente estructura, y que implemente el entrenamiento y la
#  clasificación como se ha explicado.

# class Clasificador_RL_OvR():

#     def __init__(self,class_clasif,clases):

#        .....
#     def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):

#        .....

#     def clasifica(self,ej):

#        .....

#  Excepto "class_clasif", los restantes parámetros de los métodos significan
#  lo mismo que en el apartado anterior, excepto que ahora "clases" puede ser
#  una lista con más de dos elementos. El parámetro class_clasif es el nombre
#  de la clase que implementa el clasificador binario a partir del cual se
#  forma el clasificador multiclase.

#  Un ejemplo de sesión, con el problema del iris:

# ---------------------------------------------------------------
# In [28]: from iris import *

# In [29]: iris_clases=["Iris-setosa","Iris-virginica","Iris-versicolor"]

# Creamos el clasificador, a partir de RL binaria estocástico:
# In [30]: clas_rlml1=Clasificador_RL_OvR(Clasificador_RL_ML_St,iris_clases)

# Lo entrenamos:
# In [32]: clas_rlml1.entrena(iris_entr,iris_entr_clas,100,rate_decay=True,rate=0.01)

# Clasificamos un par de ejemplos, comparándolo con su clase real:
# In [33]: clas_rlml1.clasifica(iris_entr[25]),iris_entr_clas[25]
# Out[33]: ('Iris-setosa', 'Iris-setosa')

# In [34]: clas_rlml1.clasifica(iris_entr[78]),iris_entr_clas[78]
# Out[34]: ('Iris-versicolor', 'Iris-versicolor')
# ----------------------------------------------------------------

class Clasificador_RL_OvR():
        def __init__(self,class_clasif,clases):
            self.clases = clases
            self.class_clasif = class_clasif

        def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):
            if not hasattr(self,'clasif_binarios'):
                self.clasif_binarios = {}

                for clase in self.clases:
                    clasificador_clase = self.class_clasif([0,1])

                    #A la hora de entrenar cada clasificador, aplicaremos el One versus Rest aquí, cuando le pasamos el conjunto
                    # de entrenamiento. La clase 1 del clasificador se queda igual, y las demás son clase 0.
                    clasificador_clase.entrena(entr,[0 if x!= clase else 1 for x in clas_entr], n_epochs,rate=rate,rate_decay=rate_decay)
                    self.clasif_binarios[clase]=clasificador_clase

            else:
                for clase in self.clasif_binarios:
                    self.clasif_binarios[clase].entrena(entr,[0 if x!= clase else 1 for x in clas_entr], n_epochs,rate=rate,rate_decay=rate_decay)

        def clasifica(self,ej):
            if not hasattr(self,'clasif_binarios'):
                raise ClasificadorNoEntrenado("Clasificador no entrenado")
            return self.clases[np.argmax([self.clasif_binarios[i].clasifica_prob(ej) for i in self.clases])]






# ------------------------------------------------
# II.1 Regresión logística con softmax
# ------------------------------------------------


#  Se pide igualmente implementar un clasificador en python que implemente la
#  regresión multinomial logística mdiante softmax, tal y como se describe en
#  el tema 5, pero solo la versión ESTOCÁSTICA.

#  En concreto, se pide implementar una clase python Clasificador_RL_Softmax
#  con la siguiente estructura, y que implemente el entrenamiento y la
#  clasificación como seexplica en el tema 5:

# class Clasificador_RL_Softmax():

#     def __init__(self,clases):

#        .....
#     def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):

#        .....

#     def clasifica(self,ej):

#        .....


class Clasificador_RL_Softmax():
    def __init__(self,clases):
        self.clases = clases

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):
        if type(entr[0]) is list:
            entr = [np.array(x) for x in entr]

        rate_0 = rate
        random_index_list = list(range(len(entr)))

        if not hasattr(self,'pesos_clase'):
            self.pesos_clase = {}
            for c in self.clases:
                self.pesos_clase[c]=np.random.uniform(-1,1,entr[0].size)

        for e in range(n_epochs):
            if(e%5==0):
                print("Epoch {}".format(e))
            for c in self.clases:
                random.shuffle(random_index_list)
                for i in random_index_list:
                    o_i = np.exp(sum(self.pesos_clase[c]*entr[i]))/sum(np.exp(sum(p*entr[i])) for p in self.pesos_clase.values())
                    c_y = 1 if clas_entr[i] == c else 0
                    for j in range(entr[i].size):
                        self.pesos_clase[c][j] += rate*(c_y - o_i)*entr[i][j]
                    if(rate_decay and n!=0):
                        rate = rate_0 + (2/n**(1.5))


    def clasifica(self,ej):
        if not hasattr(self,'pesos_clase'):
            raise ClasificadorNoEntrenado("Clasificador no entrenado")
        return self.clases[np.argmax(np.array(self.softmax(ej)))]

    def softmax(self,ej):
        return [np.exp(sum(self.pesos_clase[c]*ej))/sum(np.exp(sum(self.pesos_clase[p]*ej)) for p in self.clases) for c in self.clases]


# ===========================================
# PARTE III: APLICACIÓN DE LOS CLASIFICADORES
# ===========================================

# En este apartado se pide aplicar alguno de los clasificadores implementados
# en el apartado anterior,para tratar de resolver tres problemas: el de los
# votos, el de los dígitos y un tercer problema que hay que buscar.

# -------------------------------------
# III.1 Implementación del rendimiento
# -------------------------------------

# Una vez que hemos entrenado un clasificador, podemos medir su rendimiento
# sobre un conjunto de ejemplos de los que se conoce su clasificación,
# mediante el porcentaje de ejemplos clasificados correctamente. Se ide
# definir una función rendimiento(clf,X,Y) que calcula el rendimiento de
# clasificador concreto clf, sobre un conjunto de datos X cuya clasificación
# conocida viene dada por la lista Y.
# NOTA: clf es un objeto de las clases definidas en
# los apartados anteriores, que además debe estar ya entrenado.


# Por ejemplo (conectando con el ejemplo anterior):

# ---------------------------------------------------------
# In [36]: rendimiento(clas_rlml1,iris_entr,iris_entr_clas)
# Out[36]: 0.9666666666666667
# ---------------------------------------------------------

def rendimiento(clf,X,Y):
    return sum(clf.clasifica(x) == y for x,y in zip(X,Y))/len(Y)


# ----------------------------------
# III.2 Aplicando los clasificadores
# ----------------------------------

#  Obtener un clasificador para cada uno de los siguientes problemas,
#  intentando que el rendimiento obtenido sobre un conjunto independiente de
#  ejemplos de prueba sea lo mejor posible.

#  - Predecir el partido de un congresista en función de lo que ha votado en
#    las sucesivas votaciones, a partir de los datos en el archivo votos.py que
#    se suministra.

## ANOTAREMOS LOS RESULTADOS EN ESTOS COMENTARIOS:

# Perceptrón, con entrenamiento (votos.votos_entr,votos.votos_entr_clas,500) : 0.9855072463768116 rendimiento

# Regresor logístico batch, minimizando L2, con entrenamiento (votos.votos_entr,votos.votos_entr_clas,500): 0.927536231884058 rendimiento
# Regresor logístico batch, minimizando L2, con entrenamiento (votos.votos_entr,votos.votos_entr_clas,300,rate=0.01,rate_decay=True): 0.927536231884058 rendimiento
# En todas las pruebas, este clasificador obtiene el mismo valor de rendimiento, 92,8%

# Regresor logístico estocástico, minimizando L2, con entrenamiento (votos.votos_entr,votos.votos_entr_clas,500): 0.971 rendimiento
# Regresor logístico estocástico, minimizando L2, con entrenamiento (votos.votos_entr,votos.votos_entr_clas,300,rate=0.01,rate_decay=True): 0.9855 rendimiento
# Como podemos observar, este clasificador trabaja con el mismo valor de rendimiento sobre el conjunto de validación.

# Regresor logístico batch, maximizando verosimilitud, con entrenamiento (votos.votos_entr,votos.votos_entr_clas,200): 0.9275 rendimiento
# Regresor logístico batch, maximizando verosimilitud, con entrenamiento (votos.votos_entr,votos.votos_entr_clas,200,rate=0.005,rate_decay=True): 0.9275 rendimiento
# Incluso con más iteraciones de entrenamiento, se observa que el rendimiento converge al valor de 0.9275

# Regresor logístico estocástico, maximizando verosimilitud, con entrenamiento (votos.votos_entr,votos.votos_entr_clas,300): 0.971 rendimiento
# Regresor logístico estocástico, maximizando verosimilitud, con entrenamiento (votos.votos_entr,votos.votos_entr_clas,300,rate=0.005,rate_decay=True): 0.9855 rendimiento
# Observamos que, al igual que en la regresión logística minimizando L2, la versión estocástica funciona sensiblemente mejor que la versión batch.
# A su vez, observamos una lijera mejora en las versiones en las que aplicamos un learning rate más bajo con rate decay.

import votos

#### El clasificador seleccionado para medir el rendimiento sobre el conjunto de test será el Clasificador_RL_ML_St, con el siguiente entrenamiento:
##### - votos_entr y votos_entr_clases como dataset
##### - 300 epochs de entrenamiento
##### - learning rate de 0.005
##### - aplicando rate decay

rlve = Clasificador_RL_ML_St(votos.votos_clases)
rlve.entrena(votos.votos_entr,votos.votos_entr_clas,300,rate=0.005,rate_decay=True)

##### Midamos el rendimiento del clasificador sobre el conjunto votos_test:
rendimiento(rlve,votos.votos_test,votos.votos_test_clas)
## 0.9080459770114943

#  - Predecir el dígito que se ha escrito a mano y que se dispone en forma de
#    imagen pixelada, a partir de los datos que están en el archivo digidata.zip
#    que se suministra.  Cada imagen viene dada por 28x28 píxeles, y cada pixel
#    vendrá representado por un caracter "espacio en blanco" (pixel blanco) o
#    los caracteres "+" (borde del dígito) o "#" (interior del dígito). En
#    nuestro caso trataremos ambos como un pixel negro (es decir, no
#    distinguiremos entre el borde y el interior). En cada conjunto las imágenes
#    vienen todas seguidas en un fichero de texto, y las clasificaciones de cada
#    imagen (es decir, el número que representan) vienen en un fichero aparte,
#    en el mismo orden. Será necesario, por tanto, definir funciones python que
#    lean esos ficheros y obtengan los datos en el mismo formato python en el
#    que los necesitan los algoritmos.

## Funciones para parsear los ficheros de texto a datos que puedan usar los algoritmos

def digit_parser(path):
    file = open(path)
    digits=[]
    digit=[]
    idx=0
    for line in file:
        digit+=[char_converter(x) for x in line]
        idx+=1
        if(idx%28==0):
            digits.append(np.array(digit))
            digit=[]
    return digits

def char_converter(c):
    return{' ':0,'\n':0,'+':1,'#':1}[c]

def label_parser(path):
    file = open(path)
    labels=[]
    for line in file:
        labels+=[int(x) for x in line if x!='\n']
    return np.array(labels)

## RESULTADOS. En general los modelos tardan mucho en entrenar: alrededor de 2 minutos cada 5 epochs

# Regresión softmax, con entrenamiento (dataset,data_labels,10,rate=0.05): 0.86 rendimiento

# Clasificador OvR, con regresión logística minimizando L2, mismo entrenamiento: 0.592

# Elegiremos el clasificador softmax, por tener un mejor rendimiento en la fase de validación, para evaluarlo contra el test
softmnist = Clasificador_RL_Softmax(list(range(10)))

#PARA ENTRENAR (PUEDE TARDAR)
#softmnist.entrena(dataset,data_labels,10,rate=0.05)

## CAMBIAR LA RUTA SI ES NECESARIO
#test_data = digit_parser("digitdata/testimages")
#test_labels = label_parser("digitdata/testlabels")

# PARA MEDIR EL RENDIMIENTO UNA VEZ ENTRENADO
#rendimiento(softmnist,test_data,test_labels)

# RENDIMIENTO OBTENIDO: 1.0
## Como vemos, a pesar de que fallaba en la fase de validación, ha acertado todos los elementos del conjunto de test

#  - Cualquier otro problema de clasificación (por ejemplo,
#    alguno de los que se pueden encontrar en UCI Machine Learning repository,
#    http://archive.ics.uci.edu/ml/). Téngase en cuenta que el conjunto de
#    datos que se use ha de tener sus atríbutos numéricos. Sin embargo,
#    también es posible transformar atributos no numéricos en numéricos usando
#    la técnica conocida como "one hot encoding".


#  Nótese que en cualquiera de los tres casos, consiste en encontrar el
#  clasificador adecuado, entrenado con los parámetros y opciones
#  adecuadas. El entrenamiento ha de realizarse sobre el conjunto de
#  entrenamiento, y el conjunto de validación se emplea para medir el
#  rendimiento obtenido con los distintas combinaciones de parámetros y
#  opciones con las que se experimente. Finalmente, una vez elegido la mejor
#  combinación de parámetros y opciones, se da el rendimiento final sobre el
#  conjunto de test. Es importante no usar el conjunto de test para decididir
#  sobre los parámetros, sino sólo para dar el rendimiento final.

#  En nuestro caso concreto, estas son las opciones y parámetros con los que
#  hay que experimentar:

#  - En primer lugar, el tipo de clasificador usado (si es batch o
#    estaocástico, si es basado en error cuadrático o en verosimilitud, si es
#    softmax o OvR,...)
#  - n_epochs: el número de epochs realizados influye en el tiempo de
#    entrenamiento y evidentemente también en la calidad del clasificador
#    obtenido. Con un número bajo de epochs, no hay suficiente entrenamiento,
#    pero también hay que decir que un número excesivo de epochs puede
#    provocar un sobreajuste no deseado.
#  - El valor de "rate" usado.
#  - Si se usa "rate_decay" o no.
#  - Si se usa normalización o no.

# Se pide describir brevemente el proceso de experimentación en cada uno de
# los casos, y finalmente dar el clasificador con el que se obtienen mejor
# rendimiento sobre el conjunto de test correspondiente.

# Por dar una referencia, se pueden obtener clasificadores para el problema de
# los votos con un rendimiento sobre el test mayor al 90%, y para los dígitos
# un rendimiento superior al 80%.
