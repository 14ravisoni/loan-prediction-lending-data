from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html', print_result=0)


@app.route('/result', methods=['GET', 'POST'])
def result():
    issue_d = request.form['issue_d']
    data = {
        'loan_amnt': float(request.form['loan_amnt']),
        'term': int(request.form['term']),
        'int_rate': float(request.form['int_rate']),
        'sub_grade': int(request.form['sub_grade']),
        'emp_title': request.form['emp_title'],
        'emp_length': float(request.form['emp_length']),
        'annual_inc': float(request.form['annual_inc']),
        'home_ownership': int(request.form['home_ownership']),
        'verification_status': int(request.form['verification_status']),
        'pymnt_plan': int(request.form['pymnt_plan']),
        'purpose': request.form['purpose'],
        'dti': float(request.form['dti']),
        'pub_rec': int(request.form['pub_rec']),
        'application_type': int(request.form['application_type']),
        'addr_state': request.form['addr_state'],
        'tot_cur_bal': int(request.form['tot_cur_bal']),
        'num_sats': int(request.form['num_sats']),
        'total_bc_limit': float(request.form['total_bc_limit']),
        'credit_line_ratio': float(request.form['credit_line_ratio']),
        'fico_avg_score': float(request.form['fico_avg_score']),
        'disbursement_method': int(request.form['disbursement_method']),
        'issue_d_month': int(issue_d[:4]),
        'issue_d_year': issue_d[5:7]
    }
    df = pd.DataFrame(data=data, index=[0])

    # ENCODING..
    import category_encoders
    bin_encod_title = pickle.load(open('models/encoding/bin_encod_title.pkl', 'rb'))
    df_emp_title = bin_encod_title.transform(df['emp_title'].reshape(1, -1))
    df = pd.concat([df, df_emp_title], axis=1)

    bin_encod_purpose = pickle.load(open('models/encoding/bin_encod_purpose.pkl', 'rb'))
    df_purpose = bin_encod_purpose.transform(df['purpose'].reshape(1, -1))
    df = pd.concat([df, df_purpose], axis=1)

    bin_encod_addr_state = pickle.load(open('models/encoding/bin_encod_addr_state.pkl', 'rb'))
    df_addr_state = bin_encod_addr_state.transform(df['addr_state'].reshape(1, -1))
    df = pd.concat([df, df_addr_state], axis=1)

    df.drop(['emp_title', 'purpose', 'addr_state'], axis=1, inplace=True)

    # SCLAING
    from sklearn.preprocessing import MinMaxScaler
    scaler_num_cols = pickle.load(open('models/scaling/scaler_num_cols.pkl', 'rb'))
    cat_cols = []
    num_cols = []

    for column_name in df:
        df[column_name] = df[column_name].astype('float64')

    # Linear Regression
    from sklearn.linear_model import LogisticRegression
    linear_regression = pickle.load(open('models/models/model_linear_regression.pkl', 'rb'))
    y_pred = linear_regression.predict(df.reshape(1, -1))

    return render_template('prediction.html', print_result=1, result=y_pred)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    return render_template('profile.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


# Function to get categorical and numerical columns
def split_num_cat_cols(df):


    return num_cols, cat_cols
