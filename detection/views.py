from django.shortcuts import render

# Create your views here.
import pandas as pd
from .forms import UploadFileForm
from joblib import load
import os


def base(request):
    return render(request,'index.html')


# Load the trained model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'robust_scaler.joblib')
clf = load(model_path)
rob_scaler = load(scaler_path)



def preprocess_and_predict(new_data):
    new_data = pd.DataFrame(new_data)
    amount_time_data = new_data[["Amount", "Time"]]
    scaled_amount_time = rob_scaler.transform(amount_time_data)
    new_data['scaled_amount'] = scaled_amount_time[:, 0]
    new_data['scaled_time'] = scaled_amount_time[:, 1]
    new_data.drop(['Amount', 'Time'], axis=1, inplace=True)
    new_data = new_data[['scaled_amount', 'scaled_time'] + [col for col in new_data.columns if col not in ['scaled_amount', 'scaled_time']]]
    predictions = clf.predict(new_data)
    return predictions,new_data

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            data = pd.read_csv(file, delimiter=';')
            predictions,processed_data = preprocess_and_predict(data)
            processed_data['Prediction'] = predictions
            return render(request, 'result.html', {'predictions': predictions, 'data':processed_data})
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})
