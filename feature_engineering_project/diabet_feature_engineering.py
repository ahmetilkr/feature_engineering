import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Qt5Agg')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', None)


def dataset_load():
    df=pd.read_csv("../diabetes-230330-150402/diabetes/diabetes.csv")
    return df


df=dataset_load()

df.shape

df.head()

df.info()

df.isnull().sum()

df.nunique()

quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
df.describe(quantiles).T



def grab_col_names(dataframe,cat_th=10,car_th=20):
    cat_col=[col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_col=[col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_but_cat=[col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car=[col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]

    num_col=[col for col in num_col if col not in num_but_cat]

    cat_col = cat_col + num_but_cat
    cat_col=[col for col in cat_col if col not in cat_but_car]



    print(f"Observations={dataframe.shape[0]}")
    print(f"Variables={dataframe.shape[1]}")
    print(f"cat_cols=:{len(cat_col)} tane = {cat_col}")
    print(f"num_cols=:{len(num_col)} tane = {num_col}")
    print(f"cat_but_car:{len(cat_but_car)} tane = {cat_but_car}")
    print(f"num_but_cat:{len(num_but_cat)} tane = {num_but_cat}")

    return cat_col,num_col,num_but_cat,cat_but_car


cat_col,num_col,num_but_cat,cat_but_car= grab_col_names(df,cat_th=10,car_th=20)


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({"value_count": dataframe[col_name].value_counts(),
                  "ratio": dataframe[col_name].value_counts() / dataframe.shape[0] * 100}))
    print("######################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe, palette="viridis")   #kategorik verileri sayar
        plt.show(block=True)


cat_summary(df, "Outcome" , plot=True)


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################


def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)   #hist nümerik verileri sayar
        # bins=20, histogram grafiğinde sayısal verinin en küçük değeri ile en büyük değeri arasındaki mesafenin 20 eşit parçaya (aralığa/kutuya) bölüneceğini belirtir.
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_col:
    num_summary(df, col, plot=True)


##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################


#Adım 4:Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
#numerik değişkenlerin ortalaması)

def target_summary_with_num(dataframe,col,target):
    print(df.groupby(target)[col].mean())

for col in num_col:
    target_summary_with_num(df,col,"Outcome")


def cat_summary_with_target(dataframe, col, target):
    print(df.groupby(col)[target].mean())


for col in cat_col:
    target_summary_with_num(df, col, "Outcome")


##################################
# KORELASYON
##################################



##################################
# EKSİK DEĞER ANALİZİ
##################################

zero_cols=[col for col in df.columns if ((df[col] == 0).any() and col not in ["Outcome","Pregnancies"])]

for col in zero_cols:
    df[col] = np.where((df[col] == 0),np.nan,df[col])


df.isnull().sum()


def missing_values_table(dataframe):

    na_cols=[col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    miss_count=dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio=(dataframe[na_cols].isnull().sum() * 100 / len(df)).sort_values(ascending=False)
    missing_df=pd.concat([miss_count,ratio],axis=1,keys=["count","ratio"])
    print(missing_df)
    return na_cols


na_cols=missing_values_table(df)


# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi

def missing_vs_target(dataframe , na_cols_, target):
    temp_df = dataframe.copy()
    for col in na_cols_:
        temp_df[col + "_NA_FLAG"]=np.where(temp_df[col].isnull(),1,0)

    na_flag_cols = temp_df.columns[temp_df.columns.str.contains("_NA_FLAG")].tolist()
    for col in na_flag_cols:
        print(pd.DataFrame({"count":temp_df.groupby(col)[target].count(),
                      "target_mean":temp_df.groupby(col)[target].mean()}))



missing_vs_target(df, na_cols,"Outcome")

for col in zero_cols:
    df.loc[df[col].isnull(), col] = df[col].median()

##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_threshold(dataframe_,col,q1=0.05,q3=0.95):
    quartile3=dataframe_[col].quantile(q3)
    quartile1=dataframe_[col].quantile(q1)
    IQR=quartile3-quartile1
    up_limit=quartile3+1.5*IQR
    low_limit=quartile1-1.5*IQR

    return up_limit,low_limit


def check_outlier(dataframe_,col_):
    up_limit,low_limit=outlier_threshold(dataframe_, col_, q1=0.05, q3=0.95)
    if dataframe_[(dataframe_[col_] > up_limit) | (dataframe_[col_] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe_,col_):
        dataframe_[col_] = dataframe_[col_].astype(float)
        up_limit,low_limit=outlier_threshold(dataframe_,col_)
        if check_outlier(dataframe_,col_):
            dataframe_.loc[dataframe_[col] > up_limit , col] = up_limit
            dataframe_.loc[dataframe_[col] < low_limit, col] = low_limit


for col in num_col:
    print(col,check_outlier(df,col))
    replace_with_thresholds(df,col)



##################################
# ÖZELLİK ÇIKARIMI
##################################

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Glukoz degerini kategorik değişkene çevirme
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma 3 kırılım yakalandı.
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# df["NEW_AGE_CAT"].head() + "_" + df["NEW_BMI"].head().astype(str)

# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)


df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

df.head()

df.columns = [col.upper() for col in df.columns]

##################################
# ENCODING
##################################

cat_col,num_col,num_but_cat,cat_but_car= grab_col_names(df,cat_th=10,car_th=20)

def label_encoder(dataframe, binary_col_):
    labelencoder = LabelEncoder()
    dataframe[binary_col_]=labelencoder.fit_transform(dataframe[binary_col_])
    return dataframe


binary_cols=[col for col in df.columns if df[col].dtype=="O" and df[col].nunique() == 2]

for col in binary_cols:
    df=label_encoder(df, col)



def one_hot_encoder(dataframe,cat_cols):
    dataframe=pd.get_dummies(dataframe,columns=cat_cols,drop_first=True, dtype=int)
    return dataframe


cat_cols=[col for col in cat_col if col not in binary_cols and col not in ["OUTCOME"]]

df=one_hot_encoder(df,cat_cols)


##################################
# AYIRMA & ÖLÇEKLEME
##################################

X=df.drop("OUTCOME",axis=1)
Y=df["OUTCOME"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=17)
X_train.head()

_,X_train_num_col,_,_=grab_col_names(X_train)

scaler=StandardScaler()
X_train[X_train_num_col]=scaler.fit_transform(X_train[X_train_num_col])
X_test[X_train_num_col]=scaler.transform(X_test[X_train_num_col])

randomc_model = RandomForestClassifier(random_state=46).fit(X_train, Y_train)

y_predict=randomc_model.predict(X_test)


print(f"Accuracy: {accuracy_score(Y_test,y_predict)}")
print(f"Recall: {recall_score(Y_test,y_predict)}")
print(f"Precision: {precision_score(Y_test,y_predict)}")
print(f"F1: {f1_score(Y_test,y_predict)}")
print(f"Auc: {roc_auc_score(Y_test,y_predict)}")


##################################
# FEATURE IMPORTANCE
##################################


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num], palette="viridis")
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(randomc_model, X)


