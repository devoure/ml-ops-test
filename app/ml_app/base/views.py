from django.shortcuts import render, redirect
from .utils import predict

# Create your views here.
def index(request):
    if request.method == 'POST':
        values = []
        wins = request.POST.get('wins')
        draws = request.POST.get('draws')
        gd = request.POST.get('gd')
        values.extend([wins, draws, gd])

        prediction = predict(values) 
        request.session['pred'] = {'wins':wins,
                                   'draws':draws,
                                   'gd':gd,
                                   'result':str(prediction[0])}

        return redirect('prediction')

    return render(request, 'home.html', {})

def get_prediction(request):
    pred = request.session.get('pred') 
    return render(request, 'prediction.html', {'pred':pred})
