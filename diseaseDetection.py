import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

diseases = {0: 'Tomato Late blight',
            1: 'Tomato healthy',
            2: 'Tomato Early blight',
            3: 'Tomato Septoria leaf spot',
            4: 'Tomato Tomato Yellow Leaf Curl Virus',
            5: 'Tomato Bacterial spot',
            6: 'Tomato Target Spot',
            7: 'Tomato Tomato mosaic virus',
            8: 'Tomato Leaf Mold',
            9: 'Tomato Spider mites Two-spotted spider mite'}

def predictDisease(data):
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    val = pickled_model.predict([data])
    return diseases[int(val)]