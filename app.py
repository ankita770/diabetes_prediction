from flask import Flask,render_template,request,redirect
import pickle
import numpy as np

app = Flask(__name__,template_folder='template')

model=pickle.load(open('model.pkl','rb'))
@app.route('/', methods=['GET','POST'])
def hello():
    return render_template("base.html")

@app.route('/predict',methods=['POST'])
def predict():
    prg = request.form['prg']
    glc = request.form['glc']
    bp = request.form['bp']
    skt = request.form['skt']
    ins = request.form['ins']
    bmi = request.form['bmi']
    ped = request.form['ped']
    age = request.form['age']

    prg=int(prg)
    glc=int(glc)
    bp=int(bp)
    skt=int(skt)
    ins=int(ins)
    bmi=float(bmi)
    ped=float(ped)
    age=int(age)

    final_features = np.array([(prg, glc, bp, skt, ins, bmi, ped, age)])
    prediction = model.predict(final_features)

    return render_template("base.html", prediction_text='The patient has diabetes: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
    
    

