import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import joblib as jb

app=Flask(__name__,static_url_path='/static')

@app.route('/')
def home():
    return render_template('land.htm')

@app.route('/prediksi', methods=['POST'])
def predict():
    try:
        tweet=str(request.form['tweet']).lower()
        
        prediksi=model.predict([tweet])[0]
        probabilitas=model.predict_proba([tweet])
        enumprob=list(enumerate(probabilitas[0]))
        enumprob.sort(key=lambda x:x[1],reverse=True)
        probutama=enumprob[:5]

        df=pd.read_csv('Twitter_Emotion_Dataset.csv')
        sortedlist=df['label'].unique()
        sortedlist.sort()
        
        plotkategori=[]
        plotprob=[]
        for i in probutama:
            plotkategori.append(sortedlist[i[0]])
            plotprob.append(i[1]*100)

        plt.close()
        sns.set(style="darkgrid")
        sns.set_context("talk")
        ax=sns.barplot(plotprob,plotkategori,palette="Blues_d")
        ax.set(xlabel='Probability (%)',ylabel='')
        plotlist=[(plotprob[i],plotkategori[i]) for i in range(0,len(plotprob))]
        xticks=np.arange(0,101,20)
        index=0
        for a,b in plotlist:
            ax.text(a+13.5,index+0.1,str(round(a,2))+'%',color='black',ha="center")
            index+=1
        ax.set_xticks(xticks)
        plt.tight_layout()
        fig=ax.get_figure()
        
        img=io.BytesIO()
        fig.savefig(img,format='png',transparent=True)
        img.seek(0)
        graph_url=base64.b64encode(img.getvalue()).decode()
        graph='data:image/png;base64,{}'.format(graph_url)

        statement="We also have the sentiment tendency for you. Here's the visualization: "
        
        predictData=[prediksi,statement,graph]
        return render_template('predict.htm',prediction=predictData)
    except:
        return render_template('error.htm') 

@app.route('/NotFound')
def notFound():
    return render_template('error.htm')

@app.errorhandler(404)
def notFound404(error):
    return render_template('error.htm')

if __name__=='__main__':
    model=jb.load('modelComplement')
    app.run(debug=True)