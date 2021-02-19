from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import category_encoders as ce
import pickle

app = Flask(__name__)


# Function to get categorical and numerical columns
def split_num_cat_cols(df):
    cat_cols = []
    num_cols = []

    for column_name in df:
        if df[column_name].dtype == 'int64' or df[column_name].dtype == 'float64':
            num_cols.append(column_name)
        else:
            cat_cols.append(column_name)

    return num_cols, cat_cols


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
        'loan_amnt': request.form['loan_amnt'],
        'term': request.form['term'],
        'int_rate': request.form['int_rate'],
        'sub_grade': request.form['sub_grade'],
        'emp_title': request.form['emp_title'],
        'emp_length': request.form['emp_length'],
        'annual_inc': request.form['annual_inc'],
        'home_ownership': request.form['home_ownership'],
        'verification_status': request.form['verification_status'],
        'pymnt_plan': request.form['pymnt_plan'],
        'purpose': request.form['purpose'],
        'dti': request.form['dti'],
        'pub_rec': request.form['pub_rec'],
        'application_type': request.form['application_type'],
        'addr_state': request.form['addr_state'],
        'tot_cur_bal': request.form['tot_cur_bal'],
        'num_sats': request.form['num_sats'],
        'total_bc_limit': request.form['total_bc_limit'],
        'credit_line_ratio': request.form['credit_line_ratio'],
        'fico_avg_score': request.form['fico_avg_score'],
        'disbursement_method': request.form['disbursement_method'],
        'issue_d_month': issue_d[:4],
        'issue_d_year': issue_d[5:7]
    }
    df = pd.DataFrame(data=data, index=[0])

    # ENCODING..
    bin_encod_title = pickle.load(open('models/encoding/bin_encod_title.pkl', 'rb'))
    df_emp_title = bin_encod_title.fit(df['emp_title'])
    df = pd.concat([df, df_emp_title], axis=1)

    bin_encod_purpose = pickle.load(open('models/encoding/bin_encod_purpose.pkl', 'rb'))
    df_purpose = bin_encod_purpose.fit_transform(df['purpose'])
    df = pd.concat([df, df_purpose], axis=1)

    bin_encod_addr_state = pickle.load(open('models/encoding/bin_encod_addr_state.pkl', 'rb'))
    df_addr_state = bin_encod_addr_state.fit_transform(df['addr_state'])
    df = pd.concat([df, df_addr_state], axis=1)

    df.drop(['emp_title', 'purpose', 'addr_state'], axis=1, inplace=True)

    # SCLAING
    scaler_num_cols = pickle.load(open('models/scaling/scaler_num_cols.pkl', 'rb'))
    num_cols, cat_cols = split_num_cat_cols(df)
    df[num_cols] = scaler_num_cols.fit(df[num_cols])

    # Linear Regression
    linear_regression = pickle.load(open('models/models/model_linear_regression.pkl', 'rb'))
    y_pred = linear_regression.predict(df)

    return render_template('prediction.html', print_result=1, result=y_pred)


@app.route('/profile')
def profile():
    return render_template('profile.html')


@app.route('/home')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

