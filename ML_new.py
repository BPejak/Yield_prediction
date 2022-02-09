import numpy as np
# from keras.layers import Dense, Activation
# from keras.models import Sequential
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import scipy
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn import tree
from sklearn.neural_network import MLPRegressor
# import plotly.graph_objects as go
#import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from scipy import stats
import seaborn as sns
# from matplotlib import pyplot
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import StackingRegressor
from sklearn.decomposition import PCA
from sklearn import decomposition

# # Importing the dataset
# file_path_sat_data = 'final_dataset_V5.xlsx'
# # file_path_sat_data ='C:\\Users\\Pejak\\PycharmProjects\\Yield_monitor\\Naweiluo\\V3\\final_dataset_V8.xlsx'
# xl = pd.ExcelFile(file_path_sat_data)
# data = xl.parse('Sheet1')
# create dataset

# 7386/60, 1879/60
# 'PC 1 core', 'PC 3 cores'
h1 = [7386/60]
b1 = ['1 core']
height = [2606/60, 1200/60, 987/60, 680/60, 677/60, 664/60]
bars = ('1 core', '3 cores', '6 cores', '9 cores', '12 cores', '15 cores')
#
# x_pos = np.arange(len(bars))+1
# x_pos1 = np.arange(len(b1))
# plt.bar(x_pos1, h1, color='c', label = 'PC')
# plt.bar(x_pos, height, color='m', label = 'HPC')
# plt.ylabel('time (minutes)')
# plt.grid(color='r',axis='y', linestyle='--')
# plt.legend()
# plt.xticks(np.concatenate([x_pos1, x_pos]), np.concatenate([b1, bars]), rotation=45)
# # plt.xticks(x_pos1, b1, rotation=45)
# # plt.xticks(x_pos, bars, rotation=45)
# plt.show()
a=1


def racunanje_indeksa(sat_data):
    # data = sat_data.drop(['Pixel_ID', 'Yield','Area','Longitude', 'Latitude', 'Parcel_ID', 'Year'], axis=1)
    data = sat_data.drop(['Pixel_ID', 'Yield', 'Longitude', 'Latitude', 'Parcel_ID', 'Year'], axis=1)
    #racunanje vegetativnih indeksa
    data = data.to_numpy()
    blue_matrica = data[:, 1::12].astype(float)
    green_matrica = data[:, 2::12].astype(float)
    red_matrica = data[:, 3::13].astype(float)
    red_edge1_matrica = data[:, 4::12].astype(float)
    red_edge2_matrica = data[:, 5::12].astype(float)
    red_edge3_matrica = data[:, 6::12].astype(float)
    nir_matrica = data[:, 7::12].astype(float)
    nir2_matrica = data[:, 8::12].astype(float)
    swir_matrica = data[:, 10::12].astype(float)
    swir2_matrica = data[:, 11::12].astype(float)

    NDVI = np.divide((nir_matrica - red_matrica), (nir_matrica + red_matrica))

    NDVIRE1 = np.divide((nir_matrica - red_edge1_matrica), (nir_matrica + red_edge1_matrica))

    NDVIRE2 = np.divide((nir_matrica - red_edge2_matrica), (nir_matrica + red_edge2_matrica))

    NDVIRE3 = np.divide((nir_matrica - red_edge3_matrica), (nir_matrica + red_edge3_matrica))

    G = 2.5
    C1 = 6.0
    C2 = 7.5   #5.0  # C2=7.5
    L = 1     # od 0 do 1, probaj 0.5
    EVI = np.divide((G * (nir_matrica - red_matrica)), (nir_matrica + C1 * red_matrica - C2 * blue_matrica + L))
    EVI2 = np.divide(((nir_matrica - red_matrica)), (nir_matrica + (2.4 * red_matrica) + L)) * G

    VARI = np.divide((green_matrica - red_matrica), (green_matrica + red_matrica - blue_matrica))

    SAVI = np.divide((nir_matrica - red_matrica), (nir_matrica + red_matrica + 0.5)) * (1 + 0.5)

    gamma = 1.7
    ARVI = np.divide((nir_matrica - (red_matrica - gamma * (blue_matrica - red_matrica))),
                     (nir_matrica + (red_matrica - gamma * (blue_matrica - red_matrica))))

    GARI = np.divide((nir_matrica - (green_matrica - gamma * (blue_matrica - red_matrica))),
                     (nir_matrica + (green_matrica - gamma * (blue_matrica - red_matrica))))

    VDVI = np.divide((2 * green_matrica - red_matrica - blue_matrica), (2 * green_matrica + red_matrica + blue_matrica))

    NDWI = np.divide((nir_matrica - swir_matrica), (nir_matrica + swir_matrica))
    NDWI2 = np.divide((green_matrica - nir_matrica), (green_matrica + nir_matrica))  # vodene povrsine

    NLI = np.divide(((nir_matrica / (2 ** 16)) ** 2 - red_matrica / (2 ** 16)),
                    ((nir_matrica / (2 ** 16)) ** 2 + red_matrica / (2 ** 16)))  # vrednosti moraju biti izmedju 0 i 1
    NLI2 = np.divide(((nir_matrica / (2 ** 16)) ** 2 - swir_matrica / (2 ** 16)),
                     ((nir_matrica / (2 ** 16)) ** 2 + swir_matrica / (2 ** 16)))  # vrednosti moraju biti izmedju 0 i 1

    L = 0.5
    MNLI = np.divide((((nir_matrica / (2 ** 16)) ** 2 - red_matrica / (2 ** 16)) * (1 + L)), ((((nir_matrica / (2 ** 16)) ** 2 + red_matrica / (2 ** 16)) + L)))  # vrednosti moraju biti izmedju 0 i 1
    MNLI2 = np.divide((((nir_matrica / (2 ** 16)) ** 2 - swir_matrica / (2 ** 16)) * (1 + L)), ((((nir_matrica / (2 ** 16)) ** 2 + swir_matrica / (2 ** 16)) + L)))  # vrednosti moraju biti izmedju 0 i 1

    NMDI = np.divide((nir2_matrica - (swir_matrica-swir2_matrica)), (nir2_matrica + (swir_matrica-swir2_matrica)))
    # panic
    TG = (green_matrica - 0.39 * red_matrica - 0.61 * blue_matrica) / 13000
    GLI = np.divide(2 * green_matrica - red_matrica - blue_matrica, 2 * green_matrica + red_matrica + blue_matrica + 1)
    ExG = 2 * green_matrica - red_matrica - blue_matrica
    CIVE = 0.441 * red_matrica - 0.811 * green_matrica + 0.385 * blue_matrica + 18.78745 # * 13000

    AWEI = 4 * (green_matrica - swir_matrica) - (0.25 * nir_matrica + 2.75 * swir2_matrica)

    GRVI = np.divide((green_matrica - red_matrica), (green_matrica + red_matrica))

    DVI = nir_matrica - red_matrica

    data_indeksi = np.concatenate((NDVI, NDVIRE1, NDVIRE2, NDVIRE3, VARI, EVI, EVI2, ARVI, GARI,
                                   SAVI, VDVI, NMDI, NDWI, NDWI2, NLI, NLI2, MNLI, MNLI2, TG, GLI, ExG, CIVE, AWEI, GRVI, DVI), axis=1)
    # data_indeksi = np.concatenate((NDVI, NDVIRE1, NDVIRE2, NDVIRE3, VARI, EVI, EVI2, GARI, SAVI, VDVI, NMDI), axis=1)
    data_indeksi = pd.DataFrame(data_indeksi)
    data_ = pd.concat([sat_data.reset_index(), data_indeksi.reset_index()],axis=1)
    return data_

file_path_sat_data = 'final_dataset_V5.xlsx' #'final_dataset_V5.xlsx'
xl = pd.ExcelFile(file_path_sat_data)
data_sat = xl.parse('Sheet1')

data_sat = data_sat.dropna() # odbacujem uzorke koji nemaju yield
data_sat = data_sat[data_sat.Yield != 0]
data_sat = data_sat[data_sat.Yield != 'None']
#
data_sat = racunanje_indeksa(data_sat)
data_sat = data_sat.dropna() # odbacujem uzorke koji nemaju yield
data_sat = data_sat[data_sat.Yield != 0]
data_sat = data_sat[data_sat.Yield != 'None']

file_path_soil_data = 'soil_dataset.xlsx'
x2 = pd.ExcelFile(file_path_soil_data)
data_soil = x2.parse('Sheet1')
data_soil = data_soil.drop(['Longitude','Latitude','Year'], axis=1)


data = pd.merge(data_sat, data_soil, on=["Parcel_ID","Pixel_ID"],how='outer')
# np.random.seed(0)
np.random.seed(25)

data = data.dropna() # odbacujem uzorke koji nemaju yield
data = data[data.Yield != 0]
data = data[data.Yield != 'None']
# data = data[(data['Yield'] > 100) & (data['Yield'] < 7000)] ########################################################
data = data[(data['Yield'] > 1) & (data['Yield'] < 70)]
# data = data[(data['Yield'] > 20) & (data['Yield'] < 50)]

# data.to_excel('FINAL_DATASET.xlsx', index=False)
# Splitting the dataset into the Training set and Test set
test_samples_list = ["18_09_10-18_17_10_ID6",
                     "18_08_10-13_37_15_ID10",
                     "18_08_10-13_37_15_ID26",
                     "18_08_10-13_37_15_ID14",
                     "18_09_10-18_17_10_ID10",
                     "19_09_15-19_10_32_ID13",
                     "19_09_21-14_12_45_ID1",
                     "19_09_22-11_53_36_ID8",
                     "20_09_21-13_22_28_ID1",
                     "20_09_21-13_22_28_ID2"
                    ]

# "19_09_15-19_10_32_ID3", # out
# "19_09_30-19_35_19_ID16", # out

###### SVE IZ 2020
# test_samples_list = ['20_09_21-13_22_28_ID1', '20_09_21-13_22_28_ID2', '20_09_21-17_57_36_ID1', '20_10_04-10_24_58_ID1',
#                      '20_10_04-10_24_58_ID2', '20_10_04-10_24_58_ID3', '20_10_04-10_24_58_ID4', '20_10_04-10_24_58_ID5',
#                      '20_10_04-10_24_58_ID6', '20_10_04-10_24_58_ID7', '20_10_04-10_24_58_ID8']

test = data[data['Parcel_ID'].isin(test_samples_list)].drop(['Parcel_ID', 'Pixel_ID'], axis=1).astype("float32")
train = data[~data['Parcel_ID'].isin(test_samples_list)].drop(['Parcel_ID', 'Pixel_ID'], axis=1).astype("float32")

y_test = test.loc[:,'Yield'].astype("float32")
y_train = train.loc[:,'Yield'].astype("float32")

# poly = PolynomialFeatures(2)

X_test = test.drop(['Longitude','Latitude','Yield','Area','Year'], axis=1)
# X_test = test.drop(['Longitude','Latitude','Yield','Year', 'index'], axis=1)
# pca = decomposition.PCA(n_components=60)
# pca.fit(X_test)
# X_test = pca.transform(X_test)
# X_test = poly.fit_transform(X_test)
X_train = train.drop(['Longitude','Latitude','Yield','Area','Year'], axis=1)
# X_train = train.drop(['Longitude','Latitude','Yield','Year', 'index'], axis=1)
# pca = decomposition.PCA(n_components=60)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_train = poly.fit_transform(X_train)

# X = data.drop(['Longitude','Latitude','Pixel_ID','Parcel_ID','Yield','Area','Year'], axis=1)
# y = data.loc[:,'Yield'].astype("float32")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.08, random_state = 0) # test_size = 0.08,

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
a=2
##### MODELS #####
# model =  tree.DecisionTreeRegressor()
model = RandomForestRegressor(n_estimators = 100)
# model = SGDRegressor()
# model = svm.SVR()
# model = xgb.XGBRegressor(n_estimators=100,random_state=1)

# estimators = [('sec_reg',xgb.XGBRegressor(n_estimators=100,random_state=1))] #,svm.SVR(), ('fourth_reg',MLPRegressor())  , ('third_reg', RandomForestRegressor(n_estimators = 100)),   SGDRegressor(random_state=1)), xgb.XGBRegressor(n_estimators=100,random_state=1))
# model = StackingRegressor(estimators=estimators, final_estimator = svm.SVR()) # (,random_state=0))

# model_1 = MLPRegressor()#(random_state=1, max_iter=500)
# model_2 = svm.SVR()
# model_3 = RandomForestRegressor(n_estimators=100) # max_depth = 5 n_estimators=300
# model_4 = xgb.XGBRegressor(n_estimators = 100)#n_estimators = 10, max_depth = 5,n_repeats=3  #objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
# model_5 = SGDRegressor()

# Fitting the model to the Training set
model.fit(X_train, y_train)
# model_1.fit(X_train, y_train)
# model_2.fit(X_train, y_train)
# model_3.fit(X_train, y_train)
# model_4.fit(X_train, y_train)
# model_5.fit(X_train, y_train)

# # get importance
# importance = model.feature_importances_
# ind = np.arange(len(importance))
# plt.bar(ind,importance)
# plt.show()

# # summarize feature importance
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# pyplot.bar([x for x in range(len(importance))], importance)
# pyplot.show()

y_pred = model.predict(X_test)
# y_pred_1 = model_1.predict(X_test)
# y_pred_2 = model_2.predict(X_test)
# y_pred_3 = model_3.predict(X_test)
# y_pred_4 = model_4.predict(X_test)
# y_pred_5 = model_5.predict(X_test)

# df_all_pred = np.concatenate([y_pred_1,y_pred_2,y_pred_3,y_pred_4,y_pred_5]).reshape(len(y_pred_1),5)
# y_pred = np.median(df_all_pred,axis=1)

MAE = mean_absolute_error(y_test, y_pred)  # Mean Abosulte Error
RMSE = sqrt(mean_squared_error(y_test, y_pred)) # Root Mean Square Error
PCC = scipy.stats.pearsonr(y_test, y_pred)  # Pearson Correlation Coefficient
SCC = scipy.stats.spearmanr(y_test, y_pred)  # Spearman rank-order Correlation Coefficient
R2 = r2_score(y_test, y_pred)
print('')
print('Root Mean Square Error: ',round(RMSE,2))
print('Mean Abosulte Error: ',round(MAE,2))
print('R2: ', round(R2,2))
print('Pearson Correlation Coefficient: ', round(PCC[0],2))
print('Spearman Correlation Coefficient:  ', round(SCC[0],2))

print('')
print(round(RMSE,2),'&',round(MAE,2),'&', round(R2,2),'&', round(PCC[0],2),'&',round(SCC[0],2))

############# PLOT OBICAN KAO SIGNAL
plt.plot(list(y_test), color = 'red', label = 'Real yield')
plt.plot(y_pred, color = 'blue', label = 'Predicted yield')

plt.title('Prediction')
plt.legend()
plt.ylabel('Yield kg/pixel')
plt.xlabel('Pixel number')
plt.show()

############# PLOT OBICAN KAO TACKICE
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Real yield [kg/pixel]')
ax.set_ylabel('Predicted yield [kg/pixel]')
plt.show()
#
############# PLOT KAO PROBABILITY
# sns.distplot( y_train,hist=False ,norm_hist=True, color="blue", label="Training")
# sns.distplot( y_test,hist=False ,norm_hist=True, color="red", label="Test")
sns.distplot( y_test,hist=False , color="blue", label="Real yield") #norm_hist=True,
sns.distplot( y_pred,hist=False , color="red", label="Predicted yield") #norm_hist=True,d
plt.legend(loc='upper right')
plt.title('Yield')
plt.xlabel('kg/pixel')
plt.ylabel("Probability")
plt.show()

############# PLOT SA KRUZICIMA PRED I REAL SA KONTRASTOM
r, p_val = stats.pearsonr(y_test, y_pred)
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5, label="data")
ax.set_xlabel(r'Real yield [kg/pixel]', fontsize=15)
ax.set_ylabel(r'Predicted yield [kg/pixel]', fontsize=15)
ax.grid(True)
fig.tight_layout()
m, b = np.polyfit(y_test, y_pred, 1)
label_line = "Fitting line " + "r = " + '{:.3f}'.format(r) + ", p_value = " + "{:.2e}".format(p_val)
plt.plot(y_test, m * y_test + b, 'r', label=label_line)
plt.legend(loc="upper left")
plt.show()
# fig.savefig("scatter_plot" + ".png", dpi=600)


scatter_axes = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=2)
x_hist_axes = plt.subplot2grid((3, 3), (0, 0), colspan=2,
                               sharex=scatter_axes)
y_hist_axes = plt.subplot2grid((3, 3), (1, 2), rowspan=2,
                               sharey=scatter_axes)

############# PLOT SA ODNOSOM REAL I PRED YIELD I HISTOGRAMIMA
scatter_axes.scatter(y_test, y_pred, alpha=0.5, label="data")
scatter_axes.set_xlabel(r'Real yield [kg/pixel]', fontsize=15)
scatter_axes.set_ylabel(r'Predicted yield [kg/pixel]', fontsize=15)
scatter_axes.grid(True)
plt.tight_layout()
m, b = np.polyfit(y_test, y_pred, 1)
label_line = "Fitting line " + "r = " + '{:.3f}'.format(r) + ", p_value = " + "{:.2e}".format(p_val)
scatter_axes.plot(y_test, m * y_test + b, 'r', label=label_line)
scatter_axes.legend(loc="upper left", fontsize=15)

x_hist_axes.hist( y_test, color='green',label="Real yield") #norm_hist=True,
x_hist_axes.legend(loc='best', fontsize=15)
x_hist_axes.set_ylabel(r'Number of pixels', fontsize=15)
#x_hist_axes.hist( y_pred, label="Predicted yield") #norm_hist=True,
# y_hist_axes.hist( y_test, label="Real yield",orientation='horizontal') #norm_hist=True,
y_hist_axes.hist( y_pred, color='orange',label="Predicted yield",orientation='horizontal') #norm_hist=True,
y_hist_axes.legend(loc='best', fontsize=15)
y_hist_axes.set_xlabel(r'Number of pixels', fontsize=15)
plt.savefig("scatter_plot" + ".png", dpi=600)
plt.show()

############# PLOT HEKSAGONI
h = sns.jointplot(y_test, y_pred, kind='hex') #kde
#sns.color_palette('magma') #dark
# JointGrid has a convenience function
h.set_axis_labels('x', 'y', fontsize=16)
# or set labels via the axes objects
h.ax_joint.set_xlabel('Real yield [kg/pixel]', fontweight='bold')
h.ax_joint.set_ylabel('Predicted yield [kg/pixel]', fontweight='bold')
# also possible to manipulate the histogram plots this way, e.g.
# h.ax_marg_y.grid('on') # with ugly consequences...
# labels appear outside of plot area, so auto-adjust
plt.tight_layout()
# plt.ylabel("Predicted yield")
# plt.xlabel("Real yield")
# plt.xlim(0, 70)
# plt.ylim(0, 70)
plt.savefig("scatter_plot_hex" + ".png", dpi=600)
plt.show()


############# ZA DIFFERENCE I STATISTIKU AREA
r, p_val = stats.pearsonr(test.loc[:,'Area'],  test.loc[:,'Yield'].astype("float32") - y_pred)
scatter_axes = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=2)
x_hist_axes = plt.subplot2grid((3, 3), (0, 0), colspan=2,sharex=scatter_axes)
y_hist_axes = plt.subplot2grid((3, 3), (1, 2), rowspan=2, sharey=scatter_axes)
scatter_axes.scatter(test.loc[:,'Area'], test.loc[:,'Yield'].astype("float32") - y_pred, alpha=0.5, label="data")
scatter_axes.set_xlabel(r'Polygon size [m^2]', fontsize=15) # pixel [100 m^2] covered by yield monitor polygon
scatter_axes.set_ylabel(r'Yield error [kg/pixel]', fontsize=15)
scatter_axes.grid(True)
plt.tight_layout()
m, b = np.polyfit(test.loc[:,'Area'],  test.loc[:,'Yield'].astype("float32") - y_pred, 1)
label_line = "Fitting line " + "r = " + '{:.3f}'.format(r) + ", p_value = " + "{:.2e}".format(p_val)
scatter_axes.plot(test.loc[:,'Area'], m * test.loc[:,'Area'] + b, 'r', label=label_line)
scatter_axes.legend(loc="upper left", fontsize=15)
x_hist_axes.hist(test.loc[:,'Area'], color='green') #norm_hist=True,
x_hist_axes.legend(loc='best', fontsize=15)
x_hist_axes.set_ylabel(r'Number of pixels', fontsize=15)
y_hist_axes.hist(test.loc[:,'Yield'].astype("float32") - y_pred, color='orange',orientation='horizontal') #norm_hist=True,
y_hist_axes.legend(loc='best', fontsize=15)
y_hist_axes.set_xlabel(r'Number of pixels', fontsize=15)
plt.show()

