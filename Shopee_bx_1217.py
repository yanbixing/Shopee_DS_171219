import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

babyraw=pd.read_csv('SKU_grouping_data/shopee_ID_data_Baby_20170820_20171120.csv')
babydesraw=pd.read_csv('SKU_grouping_data/shopee_ID_data_description_Baby_20170820_20171120.csv',engine='python')

cnt=0
deslst=[]
for itemid in babyraw['itemid']:
    cnt+=1
    if cnt%1000==0:print(cnt)
    if itemid in desrawlst: deslst.append(list(babydesraw[babydesraw['itemid']==str(itemid)]['description'])[0])
    else:deslst.append(' ')

babyraw['description']=deslst

babyraw['Nlogprice']=(np.log(np.array(list(babyraw['price'])))-11+4.3)/8

babyraw.drop(['sold','image_count','liked_count','cmt_count','rating_good','rating_normal','rating_bad','ctime','cat1_name','price_before_discount','price'],axis=1,inplace=True)

babyraw.to_csv('home_19_desmerged', sep='\t', encoding='utf-8')

#merge item model and description

ItemModelLst=[]
for unit in list(zip(list(babyraw['item_name']),list(babyraw['model_name']))):
    a,b=unit
    if pd.isnull(b):
        ItemModelLst.append(a)
    else:
        ItemModelLst.append(a+' '+b)

ItemModelDesLst=[]
for unit in list(zip(ItemModelLst,list(babyraw['description']))):
    a,b=unit
    if pd.isnull(b):
        ItemModelDesLst.append(a)
    else:
        ItemModelDesLst.append(a+' '+b)

babyraw['ItemModelDes']=ItemModelDesLst
babyraw.drop(['item_name','model_name'],axis=1,inplace=True)

lowerbrand=[]
for brand in babyraw['brand']:
    if not pd.isnull(brand):
        lowerbrand.append(brand.lower())
    else:lowerbrand.append(brand)
babyraw['brand']=lowerbrand

loweritem=[]
for item in babyraw['ItemModelDes']:
    if not pd.isnull(item):
        loweritem.append(item.lower())
    else:loweritem.append(item)
babyraw['item']=loweritem

babyraw.drop(['ItemModelDes'],axis=1,inplace=True)

brandlst=babyraw['brand']
itemlst=babyraw['item']

cleanitem=[]

for brand,item in list(zip(brandlst,itemlst)):
    if not pd.isnull(brand):
        tempitem=' '+item+' '
        tempbrand=' '+brand+' '
        item=tempitem.replace(tempbrand,' ')
        cleanitem.append(item)
    else:cleanitem.append(item)

babyraw['item']=cleanitem

# wipe the space to make vectorization correct
shopidlst=[]
for unit in babyraw['shopid']:
    shopidlst.append(str(unit))

babyraw['shopid']=shopidlst

brandlst=[]
for unit in babyraw['brand']:
    if pd.isnull(unit):
        brandlst.append('Unknown')
    else:
        brandlst.append(unit.replace(' ',''))


babyraw['brand']=brandlst

babyraw.to_csv('home_19_beforevect', sep='\t', encoding='utf-8')

itemvectorizer=TfidfVectorizer(min_df=600)
itemvectorizer.fit(babyraw['item'])
Xitem=itemvectorizer.transform(babyraw['item']).toarray()

brandvectorizer=TfidfVectorizer(min_df=600)
brandvectorizer.fit(babyraw['brand'])
Xbrand=brandvectorizer.transform(babyraw['brand']).toarray()

shopvectorizer=TfidfVectorizer(min_df=600)
shopvectorizer.fit(babyraw['shopid'])
Xshop=shopvectorizer.transform(babyraw['shopid']).toarray()


itemcodelabel=itemvectorizer.vocabulary_
brandcodelabel=brandvectorizer.vocabulary_
shopcodelabel=shopvectorizer.vocabulary_

#form matrix
X=np.concatenate((Xbrand,Xshop,Xitem),axis=1)


#vectorize cat2 and cat3

cat2labeler=LabelEncoder()
cat2labeler.fit(babyraw['cat2_name'])
cat2labeldata=cat2labeler.transform(babyraw['cat2_name'])

cat2labeldata_rh=cat2labeldata.reshape(-1,1)
cat2onehoter=OneHotEncoder()
cat2onehoter.fit(cat2labeldata_rh)
cat2onehotdata=cat2onehoter.transform(cat2labeldata_rh)

Xcat2_onehot=cat2onehotdata.toarray()


cat3labeler=LabelEncoder()
cat3labeler.fit(babyraw['cat3_name'])
cat3labeldata=cat3labeler.transform(babyraw['cat3_name'])
cat3labeldata

cat3labeldata_rh=cat3labeldata.reshape(-1,1)
cat3onehoter=OneHotEncoder()
cat3onehoter.fit(cat3labeldata_rh)
cat3onehotdata=cat3onehoter.transform(cat3labeldata_rh)

Xcat3_onehot=cat3onehotdata.toarray()

rawPriceParaLst=np.array(list(babyraw['Nlogprice']))
postPriceParaLst=[]
for unit in rawPriceParaLst:
    if unit>1:postPriceParaLst.append([1])
    elif unit<-0:postPriceParaLst.append([0])
    else: postPriceParaLst.append([unit])

Xpricepara=np.array(postPriceParaLst)

#final matrix Xtry1
Xtry1=np.concatenate((Xcat2_onehot,Xcat3_onehot,X,Xpricepara),axis=1)

corenumsf=[37]
sqerrLstf=[]
timeconsulstf=[]
#silhouettelst2=[]

BIClstf=[]

cnt=0
for cores in corenumsf:
    #####
    start_time = time.time()
    #########
    print('cores:',cores,'fitting')
    km_F=KMeans(n_clusters=cores,init='k-means++',n_init=30,max_iter=2000)#1000,0.8min/ini #2000, 0.8min/ini
    km_F.fit(Xtry1)
    print('fitting finished')
    sqerr=km_F.inertia_
    sqerrLstf.append(sqerr)
    #########
    bic=sqerr+0.5*np.log(404728)*cores*797
    BIClstf.append(bic)
    ########
    timeconsu=(time.time() - start_time)/60
    timeconsulstf.append(timeconsu)
    print("--- %s minutes ---" % timeconsu)

prediction_F=km_F.predict(Xtry1)

babyraw['prediction']=prediction_F

brandcodelabellst=list(brandcodelabel.keys())
brandcodelabellst.sort()
shopcodelabellst=list(shopcodelabel.keys())
shopcodelabellst.sort()
itemcodelabellst=list(itemcodelabel.keys())
itemcodelabellst.sort()

itemlabeler=brandcodelabellst+shopcodelabellst+itemcodelabellst

cords=km_F.cluster_centers_

# generate label through cordinates of cluster_centers_

cordid=[]
cordlb=[]

cnt=0

for cord in cords:

    corecord=list(cord)
    corecordcat2=corecord[0:len(set(babyraw['cat2_name']))]
    corecordcat3=corecord[len(set(babyraw['cat2_name'])):len(set(babyraw['cat2_name']))+len(set(babyraw['cat3_name']))]
    corecorditem=corecord[len(set(babyraw['cat2_name']))+len(set(babyraw['cat3_name'])):len(set(babyraw['cat2_name']))+len(set(babyraw['cat3_name']))+len(itemlabeler)]
    corecordnlogpricenow=corecord[len(set(babyraw['cat2_name']))+len(set(babyraw['cat3_name']))+len(itemlabeler)]

    corecordcat2maxid=corecordcat2.index(max(corecordcat2))
    corecordcat3maxid=corecordcat3.index(max(corecordcat3))
    corecorditemmaxid=corecorditem.index(max(corecorditem))
    #rawPriceParaLst=(np.array(list(babyraw['NlogPrice']))+4.3)/8
    corecordnlogprice=corecordnlogpricenow*8-4.3

    corecordcat2maxlb=cat2labeler.inverse_transform(corecordcat2maxid)
    corecordcat3maxlb=cat3labeler.inverse_transform(corecordcat3maxid)
    itemlabel=itemlabeler[corecorditemmaxid]
    #babyraw['price']=np.log(np.array(list(babyraw['price'])))
    #np.array(list(babyraw['price']))-11
    #np.exp(np.log(5))=5
    corecordprice=np.exp(corecordnlogprice+11)

    cordid.append(cnt)
    cordlb.append((corecordcat2maxlb,corecordcat3maxlb,itemlabel,corecordprice))
    cnt+=1

predname=[]

for predid in babyraw['prediction']:
    predname.append(cordlb[predid])

babyraw['predname']=predname

# generate a viewable dict form

dictdf=pd.DataFrame()
dictdf['key']=cordid
dictdf['value']=cordlb

precatid=[]
precatcat2=[]
precatcat3=[]

for preindex in set(babyraw['prediction']):
    precatid.append(preindex)
    precatcat2.append(set(babyraw[babyraw['prediction']==preindex]['cat2_name']))
    precatcat3.append(set(babyraw[babyraw['prediction']==preindex]['cat3_name']))

dictdf['NumClass']=precatid
dictdf['Cat2sum']=precatcat2
dictdf['Cat3sum']=precatcat3

#generate results csv file
resdf=pd.DataFrame()
resdf['itemid']=babyraw['itemid']
resdf['modelid']=babyraw['modelid']
resdf['group_id']=babyraw['prediction']
resdf['group_name']=babyraw['predname']
resdf.to_csv('homeres1219', sep='\t', encoding='utf-8')

dictdf.to_csv('babyraw1217_dict', sep='\t', encoding='utf-8')
