from django.shortcuts import render
from joblib import load
import os.path

model = load('./savedModels/model.joblib')


# Create your views here.
def Predictor(request):
    return render(request, 'main.html')

def formInfo(request):
    sepal_length = request.GET['sepal_length']
    sepal_width = request.GET['sepal_width']
    petal_length = request.GET['petal_length']
    petal_width = request.GET['petal_width']
    y_pred = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    print(y_pred)
    if y_pred[0] == 0:
        y_pred = "Setosa"
        img_url = "../images/setosa.jpg"

    elif y_pred[0] == 1:
        y_pred = "Verscicolor"
        img_url = "../images/versicolor.jpg"
    else:
        y_pred = "Virginica"
        img_url = "../images/virginica.jpg"

    return render(request, 'result.html', {'result': y_pred, 'img_url': img_url})