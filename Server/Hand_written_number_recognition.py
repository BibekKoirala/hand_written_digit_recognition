import matplotlib
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import os
import cv2
from flask import Flask, jsonify, request, flash, redirect
import pickle

from flask_cors import CORS
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'xcf'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

style.use('ggplot')

mnist = load_digits()
predict_index = np.random.randint(0, 1700)

X = pd.DataFrame(mnist['data'], columns=['X' + str(i) for i in range(len(mnist['data'][0]))])
y = mnist['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# clf = LinearRegression()
# clf = LogisticRegression()
# clf = SVC() # Found to be best
# clf = SVR()
# clf = SGDClassifier()
# clf = KNeighborsClassifier(n_jobs=-1)
# clf.fit(X_train, y_train)



def lin_reg(grey):
    pickle_in = open('lir.pickle', 'rb')
    clf = pickle.load(pickle_in)
    accuracy = clf.score(X_test, y_test)
    predicted = clf.predict(grey)
    return accuracy, predicted


def log_reg(grey):
    pickle_in = open('lor.pickle', 'rb')
    clf = pickle.load(pickle_in)
    accuracy = clf.score(X_test, y_test)
    predicted = clf.predict(grey)
    return accuracy, predicted


def sv_reg(grey):
    pickle_in = open('svr.pickle', 'rb')
    clf = pickle.load(pickle_in)
    accuracy = clf.score(X_test, y_test)
    predicted = clf.predict(grey)
    return accuracy, predicted


def sv_class(grey):
    pickle_in = open('svc.pickle', 'rb')
    clf = pickle.load(pickle_in)
    accuracy = clf.score(X_test, y_test)
    predicted = clf.predict(grey)
    return accuracy, predicted


def sgd_class(grey):
    pickle_in = open('sgdc.pickle', 'rb')
    clf = pickle.load(pickle_in)
    accuracy = clf.score(X_test, y_test)
    predicted = clf.predict(grey)
    return accuracy, predicted


def kn(grey):
    pickle_in = open('kn.pickle', 'rb')
    clf = pickle.load(pickle_in)
    accuracy = clf.score(X_test, y_test)
    predicted = clf.predict(grey)
    return accuracy, predicted


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return '<h1>Home Page</h1>'


@app.route('/image', methods=["POST"])
def image_pro():
    if request.method == 'POST':
        posted_data = request.files['fileToUpload']
        # data = posted_data['data']
        print(request.files)
        file = request.files['fileToUpload']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread('images/' + file.filename, cv2.IMREAD_UNCHANGED)

        print('Original Dimensions : ', img.shape)

        scale_percent = 0.741  # percent of original size
        width = int(img.shape[1] * (8 / img.shape[1]))
        height = int(img.shape[0] * (8 / img.shape[0]))
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        print('Resized Dimensions : ', resized.shape)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        grey = np.array([i // 16 for i in gray])
        grey = grey.reshape((1, -1))
        grey = 15 - grey
        print('six gray', grey.reshape((8, 8)))
        print(posted_data)

        # predict_val = prediction(grey)
        # print('Predicted value :: ', predict_val)
        algo = request.form.get('algorithm')
        if algo == 'Linear Regression':
            acc, val = lin_reg(grey)
        elif algo == 'Logistic Regression':
            acc, val = log_reg(grey)
        elif algo == 'Support vector classification':
            acc, val = sv_class(grey)
        elif algo == "Support vector Regression":
            acc, val = sv_reg(grey)
        elif algo == "Stochastic Gradient Descent":
            acc, val = sgd_class(grey)
        elif algo == "K-Nearest Neighbours":
            acc, val = kn(grey)
        return jsonify({"accuracy": str(acc), "value": str(val[0])})


if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.12')
