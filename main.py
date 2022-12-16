import pandas as pd
import sys
import urllib3
import urllib3.request as ur
import fastrank
import requests
import re
import json
#import yfinance as yf
import pyterrier as pt
import os
import numpy as np
from bs4 import BeautifulSoup

nan=np.nan
def str_to_float(row):

  l=[]
  x=[]
  if type(row)==float:
    return row
  elif row[0]=="[":
    row=row[1:-1]
    row=row.replace(" ","")
    splitted_row=row.split(",")
    for a in range(len(splitted_row)):
      if splitted_row[a]!="nan":
        l.append(float(splitted_row[a]))
        x.append(a)
    
  elif row !=nan and type(row)==str:
    cur_text=row.split("\n")[1:][:-1]
    for i in range(len(cur_text)):
      num=cur_text[i].split(" ")
      if num[-1]=="nan":
        continue
      else:
        l.append(float(num[-1]))
        x.append(i)
  row =[l,x]
  return row
def fit_finance_df(row):
      #print(row)
  if type(row)==float:
    return 0
  elif len(row[0])==0:
    return 0
  elif len(row[0])<=1:
    return 1
  elif np.polyfit(row[1],row[0],1)[0]>=0:
    return -1 #since first number represents the latest year
  else:
    return 1


def main(query):
    nan=np.nan
    # s1=pd.read_csv("finance_df_1000_2000.csv",index_col=0).loc[:1999]
    # s2=pd.read_csv("finance_df_2000_4000.csv",index_col=0).loc[:1999]
    # s3=pd.read_csv("finance_df_4000_6000.csv",index_col=0).loc[:1999]
    # s4=pd.read_csv("finance_df_6000_8000.csv",index_col=0).loc[:1999]
    # s5=pd.read_csv("finance_df_8000_plus.csv",index_col=None)
    # finance_df=pd.concat([s1,s2,s3,s4,s5],axis=0,ignore_index=True)
    

    # #data type change
    # finance_df["tse"]=finance_df["tse"].apply(str_to_float)
    # finance_df["tcl"]=finance_df["tcl"].apply(str_to_float)
    # finance_df["stable"]=finance_df["stable"].apply(str_to_float)
    # finance_df["total_cash"]=finance_df["total_cash"].apply(str_to_float)
    # finance_df["total_current_asset"]=finance_df["total_current_asset"].apply(str_to_float)
    # #fit
    # finance_df["tse_coeff"]=finance_df["tse"].apply(fit_finance_df)
    # finance_df["tcl_coeff"]=finance_df["tcl"].apply(fit_finance_df)
    # finance_df["stable_coeff"]=finance_df["stable"].apply(fit_finance_df)
    # finance_df["total_cash_coeff"]=finance_df["total_cash"].apply(fit_finance_df)
    # finance_df["total_current_asset_coeff"]=finance_df["total_current_asset"].apply(fit_finance_df)
    # finance_df_clean=finance_df.fillna(0)
    # #Calculate final order score
    # finance_df_clean["score"]=finance_df_clean["eps"]+finance_df_clean["dividend"]-finance_df_clean["tse_coeff"]-finance_df_clean["tcl_coeff"]+2*finance_df_clean["stable_coeff"]+finance_df_clean["total_cash_coeff"]+finance_df_clean["total_current_asset_coeff"]
    finance_df_clean=pd.read_csv("financial_df_clean.csv")
    new_doc=pd.read_csv('new_doc.csv')


    # t=pd.read_csv("documents_profile_clean.csv",index_col=False)
    # new_doc=pd.read_csv('documents_profile_clean.csv')
    # new_doc.rename(columns={'index': 'docno', 'longBusinessSummary': 'text'}, inplace=True)
    # new_doc=new_doc.merge(finance_df_clean,left_on="docno",right_on="ticker")
    #print(len(t))

    if not pt.started():
        pt.init()

    index_dir = './financial_bs_summary_index2'
    if not os.path.exists(index_dir + "/data.properties"):
      # create the index, using the IterDictIndexer indexer 
      indexer = pt.DFIndexer(index_dir, overwrite=True,blocks=True)
      index_ref = indexer.index(new_doc["text"], new_doc["docno"])
      index_ref.toString()
    else:
    # if you already have the index, create an IndexRef from the data in cord19_index_path
    # that we can use to load using the IndexFactory
      index_ref = pt.IndexRef.of(index_dir + "/data.properties")
      index_ref.toString()

    # indexer = pt.DFIndexer(index_dir, overwrite=True,blocks=True)
    # index_ref = indexer.index(new_doc["text"], new_doc["docno"])
    # index_ref.toString()
    index = pt.IndexFactory.of(index_ref)


    topics=pd.read_csv("topics.csv")
    topics['query']= topics['query'].str.replace(',','')

    RANK_CUTOFF = 10
    SEED=42
    from sklearn.model_selection import train_test_split
    tr_va_topics, test_topics = train_test_split(topics, test_size=10, random_state=SEED)
    train_topics, valid_topics =  train_test_split(tr_va_topics, test_size=2, random_state=SEED)

    all_query=pd.read_csv("relevance_total.csv")
    relevance=pd.read_csv("relevance_total_clean.csv")
    relevance['label']=relevance['label'].astype(int)
    #bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF")
    bm25_qe = pt.BatchRetrieve(index, wmodel="BM25", controls={"qe":"on", "qemodel" : "Bo1"})
    def tse_coeff(row):
        #qid=row["qid"]
        did=row["docno"]
        res=new_doc[new_doc['docno']==did]["stable_coeff"].values
        if len(res)==0 or res[0]<0:
            return 0#qid_doc_to_sim[(qid,did)].values[0]
        else:
            return res[0]
    pipeline=(bm25_qe) >> (
        pt.transformer.IdentityTransformer()
        **
        tfidf
        **
        pt.apply.doc_score(tse_coeff)
        # **
        # (pt.apply.doc_score(lambda row: int( row["doi"] is not None and len(row["doi"]) > 0) ))
        **
        (pt.apply.query(lambda row: 'insurance rail transportation') >> bm25_qe)#score of text for query insurance rail transportation
        **
        pt.BatchRetrieve(index, wmodel="CoordinateMatch")
    
    )
    fnames=["BM25", "TFIDF","tse_coeff" "rail_way","CoordinateMatch"]
    

    train_request = fastrank.TrainRequest.coordinate_ascent()

    params = train_request.params
    params.init_random = True
    params.normalize = True
    params.seed = 1234567
    ca_pipe = pipeline >> pt.ltr.apply_learned_model(train_request, form='fastrank')

    ca_pipe.fit(train_topics, relevance)


    ca_search_result=ca_pipe.search(query).head(20)[["docno","query"]]
    ca_search_result=ca_search_result.merge(finance_df_clean,left_on='docno',right_on='ticker')
    ca_search_result=ca_search_result.sort_values(by=['score'],ascending=False)
    print("Here are the list of companies: ",ca_search_result["docno"].to_list())
    return ca_search_result #return dataframe object

if __name__=="__main__":
    query=sys.argv[1]
    print(query)
    main(query)
