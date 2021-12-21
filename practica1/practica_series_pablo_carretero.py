#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Realizado por: Pablo Carretero Collado
# El objetivo del siguiente notebook es el análisis y la predicción de las ventas de la compañía Apple
# Para ello, se han descargado los datos de ventas desde el primer trimestre de 1990 hasta el segundo trimestre de 2021, descargados de Bloomberg


# In[3]:


# A continuación, vamos a preparar los datos para su posterior análisis
# Importamos las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


# In[4]:


# Leemos los datos, cambiando las "comas" por "puntos" para los decimales y ponemos el formato de fecha correcto
ap_df = pd.read_csv('aapl.csv', sep = ';', decimal = ",")
ap_df['Fecha'] = pd.to_datetime(ap_df['Fecha'])
ap_df = ap_df.set_index('Fecha')
ap_df.head()


# In[5]:


# Transformamos los datos en trimestrales
ap_ts = ap_df.resample("q").last()
ap_ts.tail()


# In[6]:


# Cambiamos los datos para que nos los muestre por trimestres del año natural
type('Ingresos')
ap_ts_q=ap_df['Ingresos'].astype('float64').to_period('Q').sort_index()
ap_ts_q.tail()


# In[7]:


# Graficamos los ingresos de la compañía y analizamos su estacionariedad
import seaborn as sns
sns.set(rc={'figure.figsize':(11,4)})
ax = ap_ts_q.plot(marker='o',linestyle='-')
ax.set_ylabel('Ventas de Apple')


# In[ ]:


# En el plot de ingresos de Apple se observa como carecen de estacionariedad, dado que la varianza y la media no permanecen constantes
# Esta serie temporal posee una tendencia creciente con un componente estacional especialmente notable desde el año 2013, a continuación se descompone la serie:


# In[9]:


decomposition = sm.tsa.seasonal_decompose(ap_ts['Ingresos'])
ax = decomposition.plot()


# In[8]:


# Siguiendo con el análisis, estudiamos el componente estacional.
# El siguiente gráfico muestra como las ventas de Apple son especialmente mayors en el último trimestre del año, esto se puede deber a:
# la coincidencia con grandes campañas de ventas como Navidad o "Black Friday" y también a que Apple lanza sus productos en los últimos meses del año
import statsmodels.api as sm
ax = plt.gca()
sm.graphics.tsa.quarter_plot(ap_ts_q,ax=ax)
ax.set_title('Componente estacional')


# In[10]:


from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split


# In[11]:


# Calculamos las ventas anuales de Apple
ap_year = ap_df['Ingresos'].resample("y").sum()
ap_year
# Eliminamos las ventas del último año 2021, al no estar el año cerrado
ap_year = ap_year.drop(ap_year.index[-1], axis=0)
ap_year


# In[12]:


# Dibujamos las ventas anuales de Apple
sns.set(rc={'figure.figsize':(11, 4)})
ax = ap_year.plot(marker='o', linestyle='-')
ax.set_ylabel('Ventas anuales Apple');


# In[15]:


# Calculamos el crecimiento anual de los ingresos de Apple en tanto por ciento.
# Cabe destacar que hemos eliminado los años 1989 porque al ser el primer año no se puede calcular su incremento respecto al año anterior, y
# 2021 porque no se tienen los datos de todo el año.
# También se elimina 1990 porque su variación es extrema, en comparación con el resto de años.
ap_year_crec = ap_year.pct_change().mul(100)
ap_year_crec = ap_year_crec.drop(ap_year_crec.index[0], axis=0)
ap_year_crec = ap_year_crec.drop(ap_year_crec.index[0], axis=0)
ap_year_crec = ap_year_crec.drop(ap_year_crec.index[-1], axis=0)
print(ap_year_crec)


# In[16]:


# Hacemos un plot con la variación en las ventas
sns.set(rc={'figure.figsize':(11, 4)})
ax2 = ap_year_crec.plot(marker='o', linestyle='-')
ax2.set_ylabel('Crecimiento Apple');


# In[17]:


# A continuación vamos a analizar qué modelo de predicción nos puede ofrecer mejores resultados
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split


# In[18]:


# Cogemos de conjunto de entrenamiento todos los datos menos los últimos 8 trimestres, que corresponden a la parte del Test
y_train, y_test = temporal_train_test_split(y = ap_ts ['Ingresos'].astype('float64').to_period('Q'),test_size = 8)
plot_series(y_train, y_test, labels = ["y_train","y_test"])
print(y_train.shape[0],y_test.shape[0])


# In[20]:


# Utilizamos un AutoETS para entrenar el modelo
from sktime.forecasting.ets import AutoETS


# In[21]:


# Como hemos mencionado anteriormente, especificamos que el horizonte de predicción sean los últimos 8 trimestres (1-9)
fh = np.arange(1,9)
ap_auto_model = AutoETS(auto=True,sp=4,n_jobs=-1)
ap_auto_model.fit(y_train)


# In[22]:


# Obtenemos un modelo MAM,tendencia aditiva, componente estacionario multiplicativo
print(ap_auto_model.summary())


# In[23]:


# Predecimos las ventas para las fechas del conjunto de test
ap_pred = ap_auto_model.predict(fh)
print(ap_pred)


# In[24]:


# Representamos los resultados obtenidos en las predicciones del test, junto con los datos reales
plot_series(y_train['2012':],ap_pred,y_test, labels = ['AP', 'AP PRED', 'AP REAL'])


# In[25]:


# Ahora estudiamos la precisión de las predicciones mediante el error
# MAPE
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test, ap_pred)


# In[26]:


from sktime.performance_metrics.forecasting import MeanSquaredError
#MSE
mse = MeanSquaredError()
mse(y_test, ap_pred)


# In[27]:


#RMSE
rmse = MeanSquaredError(square_root=True)
rmse(y_test, ap_pred)


# In[ ]:


# Ahora estudiamos el modelo ARIMA


# In[29]:


# Aplicamos la transformación de logaritmos
from sktime.transformations.series.boxcox import LogTransformer
transformer = LogTransformer()
log_ap_ts= transformer.fit_transform(ap_ts_q)
log_ap_ts.tail()


# In[30]:


#dibujamos la serie transformada
fig, ax =plot_series(log_ap_ts, labels=["Ventas"])
ax.set_title('Ventas Apple: Transformación LOG')


# In[31]:


# comparamos las series con y sin logaritmo
fig, ax =plot_series(ap_ts_q, labels=["Ventas"])
ax.set_title('Ventas Apple: Serie Original')
fig, ax =plot_series(log_ap_ts, labels=["Ventas"])
ax.set_title('Ventas Apple: Transformación LOG')


# In[29]:


# Graficamos
ax = log_ap_ts.plot(marker = 'o', linestyle ='-')
ax.set_ylabel('Ingresos de Apple')
ax.set_title('Ventas de Apple:Transformacion logaritmo')


# In[30]:


fig, ax =plot_series(log_ap_ts, labels=["Ventas"])
ax.set_title('Ventas Apple: Transformación LOG')


# In[32]:


# Aplicamos el modelo ARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split


# In[33]:


# Al igual que hicimos con el modelo ETS, escogemos como modelo de Test los últimos 8 trimestres, y el resto conjunto de entrenamiento
y_train, y_test = temporal_train_test_split(y =ap_ts_q, test_size=8)
log_y_train, log_y_test = temporal_train_test_split(y =log_ap_ts, test_size=8)
plot_series(y_train, y_test, labels=["y_train", "y_test"])
print(y_train.shape[0], y_test.shape[0])


# In[34]:


# Introducmos el horizonte de prediccion (8 trimestres)
fh = np.arange(len(y_test))+1
fh


# In[35]:


from sktime.forecasting.arima import AutoARIMA


# In[36]:


forecaster = AutoARIMA(sp=4,suppress_warnings=True)
forecaster.fit(log_y_train)


# In[37]:


print(forecaster.summary())


# In[38]:


# Calculamos las predicciones para el conjunto de test
log_y_pred = forecaster.predict(fh)
log_y_pred


# In[39]:


# Transformamos las predicciones de nuevo a la serie original con la exponencial
ap_predicciones_arima=np.exp(log_y_pred)
print(ap_predicciones_arima)


# In[40]:


# Vemos la precisión de las predicciones con los errores
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import MeanSquaredError


# In[41]:


# MAPE (LOGARITMOS)
mean_absolute_percentage_error(log_y_test, log_y_pred)


# In[42]:


# MAPE(ORIGINAL)
mean_absolute_percentage_error(y_test, np.exp(log_y_pred))


# In[43]:


# MSE (LOGARITMOS)
rmse = MeanSquaredError(square_root=True)
rmse(log_y_test, log_y_pred)


# In[44]:


# MSE (ORIGINAL)
rmse = MeanSquaredError(square_root=True)
rmse(y_test, np.exp(log_y_pred))


# In[45]:


# Representamos el conjunto de test, training y también los datos obtenidos con las predicciones
plot_series(y_train, np.exp(log_y_pred),y_test, labels=["AP", "AP pred", "AP REAL"])


# In[46]:


# Para observar mejor las predicciones, filtramos a partir del año 2012
plot_series(y_train["2012":], np.exp(log_y_pred),y_test, labels=["AP", "AP pred", "AP REAL"])


# In[47]:


# El modelo ARIMA ofrece unos mejores resultados, dado que el error que tiene es más bajo.
# A continuación podemos calcular las predicciones para los próximos 6 trimestres
fh = np.arange(6) + 1  # forecasting horizon
fh


# In[48]:


forecaster = AutoARIMA(sp=4,suppress_warnings=True)
forecaster.fit(log_ap_ts)


# In[49]:


print(forecaster.summary())


# In[51]:


# Obtenemos las predicciones para nuestro modelo ARIMA
log_y_pred_ap = forecaster.predict(fh)
log_y_pred_ap


# In[52]:


# De nuevo convertimos nuestras predicciones a la serie originxal mediante la exponencial
predicciones = np.exp(log_y_pred_ap)
print(predicciones)


# In[51]:


# Representamos nuestras predicciones con el resto de la serie temporal
plot_series(ap_ts_q, np.exp(log_y_pred_ap), labels=["AP", "AP pred"])


# In[52]:


# Representamos nuestras predicciones con el resto de la serie temporal, a partir de 2016
plot_series(ap_ts_q["2016":], np.exp(log_y_pred_ap), labels=["AP", "AP pred"])


# In[53]:


# A contnuación estudiamos si existió efecto COVID en el 2020. Para ello calculamos la diferencia entre los datos reales (y_test) y las predicciones de nuestro modelo para el conjunto de test (ap_predicciones_arima)
covid = y_test - ap_predicciones_arima
covid = covid.drop(covid.index[0], axis=0)
covid = covid.drop(covid.index[0], axis=0)
covid = covid.drop(covid.index[0], axis=0)
 
print(covid)
plot_series(covid) # Lo representamos en una gráfica


# In[54]:


# Lo representamos en una gráfica
sns.barplot(x=covid.index, y=covid)


# In[55]:


# A continuación, calculamos las ventas acumuladas para los años 2021 y 2022.
# Para ello utilizamos primero los años naturales:


# In[56]:


data = {'fecha': ['01/01/2021', '31/03/2021', '30/06/2021', '30/09/2021', '01/01/2022', '31/03/2022', '30/06/2022', '30/09/2022'],
        'Ingresos': [ap_ts_q.iloc[-2], ap_ts_q.iloc[-1], predicciones.iloc[0], predicciones.iloc[1], predicciones.iloc[2], predicciones.iloc[3], predicciones.iloc[4], predicciones.iloc[5]]}

predicciones2 = pd.DataFrame(data)
predicciones2


# In[56]:


# Cambiamos el formato de la fecha
predicciones2['fecha'] = pd.to_datetime(predicciones2['fecha'])
predicciones2 = predicciones2.set_index('fecha')
print(predicciones2)


# In[57]:


# Hacemos un resample del dataframe, agrupándolos por años y sumando
predicciones_anuales = predicciones2.resample('A').sum()
predicciones_anuales


# In[57]:


# Ahora lo calculamos con los años fiscales de Apple (siendo 4Q de 2020 (natural) el Q1 de 2021)
data = {'fecha': ['01/01/2021', '31/03/2021', '30/06/2021', '30/09/2021', '01/01/2022', '31/03/2022', '30/06/2022', '30/09/2022'],
        'Ingresos': [ap_ts_q.iloc[-3], ap_ts_q.iloc[-2], ap_ts_q.iloc[-1], predicciones.iloc[0], predicciones.iloc[1], predicciones.iloc[2], predicciones.iloc[3], predicciones.iloc[4]]}

predicciones_fisc = pd.DataFrame(data)
predicciones_fisc


# In[59]:


# Cambiamos el formato de las fechas
predicciones_fisc['fecha'] = pd.to_datetime(predicciones_fisc['fecha'])
predicciones_fisc = predicciones_fisc.set_index('fecha')
print(predicciones_fisc)


# In[60]:


# De nuevo hacemos un resample, agrupando por años y sumando
predicciones_fisc = predicciones_fisc.resample('A').sum()
predicciones_fisc


# In[58]:


# A continuación comparamos nuestros resultados con los del informe de Barclays
# creamos data frame de Barclays multiplicados por 1000 para que cuadre con nuestras fechas
data = {'fecha' : ['31/12/2021', '31/12/2022'],
        'Ingresos' : ['368925.0', '378619.0']}

pred_barclays = pd.DataFrame(data)
pred_barclays


# In[62]:


# Cambiamos el formato de las fechas
pred_barclays['fecha'] = pd.to_datetime(pred_barclays['fecha'])
pred_barclays = pred_barclays.set_index('fecha')
pred_barclays


# In[59]:


print(predicciones_fisc['Ingresos'])
print(pred_barclays['Ingresos']) 

