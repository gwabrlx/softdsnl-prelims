from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os

# Define the paths for the model and label encoder
model_path = os.path.join(settings.BASE_DIR, 'ml_api', 'model.pkl')
encoder_path = os.path.join(settings.BASE_DIR, 'ml_api', 'label_encoder.pkl')

 # Load the trained model and label encoder
model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

class PredictView(APIView):
    def post(self, request):
        try:
           

            # Get feature values from the request
            feature1 = float(request.data.get("feature1"))
            feature2 = float(request.data.get("feature2"))
            feature3 = float(request.data.get("feature3"))

            # Ensure we received the correct features
            if feature1 is None or feature2 is None or feature3 is None:
                return Response({"error": "Missing one or more required features"}, status=status.HTTP_400_BAD_REQUEST)

            # Prepare input data for prediction (ensure it's in a 2D array format)
            input_data = [[feature1, feature2, feature3]]

            # Make the prediction
            prediction = model.predict(input_data)

            # Decode the predicted label using label encoder
            label = label_encoder.inverse_transform(prediction)[0]

            return Response({'prediction': label})

        except Exception as e:
            # Handle any errors (e.g., missing or incorrect data)
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
