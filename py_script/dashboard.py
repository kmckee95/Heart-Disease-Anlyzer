import pickle
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
print('Loading......')

# load the saved model
rfc_saved = pickle.load(open('rfc.pickle','rb'))

full_pipeline_saved = pickle.load(open('full_pipeline.pickle','rb'))


# function to check the heart disease risk
def CheckHeartDisease(age,sex,ChestPainType,RestingBP,Cholesterol,
                      FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope):
    try:
        df_model = pd.DataFrame([],columns=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina', 'Oldpeak','ST_Slope'])

        df_model.loc[0] = [age,sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]
        
        # preprocess the person details
        X_processed = full_pipeline_saved.transform(df_model)
        
        # do the prediction
        y_pred = rfc_saved.predict(X_processed)
        
        # plot risk of heart disease based on sex
        df = pd.read_csv('heart.csv')
        target = df['HeartDisease'].replace([0,1],['Low','High'])
        data = pd.crosstab(index=df['Sex'],
                   columns=target)
        
        data.plot(kind='bar',stacked=True)
        fig1 = plt.gcf()
        plt.close()
        
        # plot count of person within given age range, with heart disease risk
        bins=[0,30,50,80]
        sns.countplot(x=pd.cut(df.Age,bins=bins),hue=target,color='r')
        fig2 = plt.gcf()
        plt.close()

        # plot graph based on ChestPainType
        sns.countplot(x=target,hue=df.ChestPainType)
        plt.xticks(np.arange(2), ['No', 'Yes']) 
        fig3 = plt.gcf()

        if y_pred[0]==0:
            return 'No Heart Disease',fig1,fig2,fig3
        else:
            return 'High Chances of Heart Disease',fig1,fig2,fig3
         
    except:
        return 'Wrong inputs',fig1,fig2,fig3

# create GUI
iface = gr.Interface(
    CheckHeartDisease, # its the function to be called with below parameters
    [
    gr.inputs.Number(label='Age (0-115)'), 
    gr.inputs.Dropdown(['M','F'],default='M'), 
    gr.inputs.Dropdown(['ATA', 'NAP', 'ASY','TA'],default='TA'),
    gr.inputs.Number(label='RESTINGBP (0-200)'), 
    gr.inputs.Number(label='CHOLESTEROL (0-603)'), 
    gr.inputs.Number(label='FASTINGBS (0-1)'),   
    gr.inputs.Dropdown(['Normal', 'ST' ,'LVH'],default='ST'),
    gr.inputs.Number(label='MAXHR (60-202)'), 

    gr.inputs.Dropdown(['Y','N'],default='Y'),
    gr.inputs.Number(label='OLDPEAK (-2.6 to 6.2)'),
    gr.inputs.Dropdown(['Up', 'Flat', 'Down'],default='Up')
    ],
    [gr.outputs.Textbox(),"plot","plot","plot"]
    
    , live=False,layout='vertical',title='Get Your Heart Disease Status',
)

iface.launch() # launch the gui
