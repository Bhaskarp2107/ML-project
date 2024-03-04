from django.shortcuts import render
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def heart_predict(request):
    if request.method == 'POST':
        bmi = float(request.POST.get('bmi'))
        smoking = float(request.POST.get('smoking'))
        alc = float(request.POST.get('alc'))
        stroke = float(request.POST.get('stroke'))
        phy = float(request.POST.get('phy'))
        men = float(request.POST.get('men'))
        diff = float(request.POST.get('diff'))
        sex = float(request.POST.get('sex'))
        age = float(request.POST.get('age'))

        path = "C:\\Users\\Bhaskar\\Desktop\\HEART\\heart_project\\heart.csv"
        data = pd.read_csv(path)

        print(data)
        print(data.info())

        le_he = LabelEncoder()
        data['heartdisease'] = le_he.fit_transform(data['HeartDisease'])

        le_se = LabelEncoder()
        data['smoking'] = le_se.fit_transform(data['Smoking'])

        le_al = LabelEncoder()
        data['alcoholdrinking'] = le_al.fit_transform(data['AlcoholDrinking'])

        le_st = LabelEncoder()
        data['stroke'] = le_st.fit_transform(data['Stroke'])

        le_sex = LabelEncoder()
        data['sex'] = le_sex.fit_transform(data['Sex'])

        le_age = LabelEncoder()
        data['age'] = le_age.fit_transform(data['AgeCategory'])

        print(data['age'])

        le_dif = LabelEncoder()
        data['diffwalking'] = le_dif.fit_transform(data['DiffWalking'])

        inputs = data[['BMI','smoking','alcoholdrinking','stroke','PhysicalHealth','MentalHealth','sex','diffwalking','age']]
        output = data['HeartDisease']

        model = DecisionTreeClassifier()
        model.fit(inputs,output)

        prediction = model.predict([[bmi,smoking,alc,stroke,phy,men,diff,sex,age]])
        return render(request,'heart_predict.html',{'prediction':prediction[0]})
    else:
        return render(request,'heart_predict.html')
