# for serving web interface
from flask import Flask, render_template, flash, redirect, request, url_for, session
from flask_session import Session
from werkzeug.utils import secure_filename
from markupsafe import Markup, escape
import re
import os

# for data handling and easy operation
import pandas as pd
import numpy as np
import seaborn as sns

# for data visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
## for classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
## for regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

app = Flask(__name__)
app.secret_key = '69'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

########################################################################################################### Variables
### Flask Variables
UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'csv'}

### Variables used throughout our code
# dataframe = None
# dataLoad = None
# dataColumns = None
# dataframe_num = None


########################################################################################################### Flask routes
# route for csv upload page
@app.route('/')
def datasetSubmit():
    if 'dataLoad' in session:
        reupload()

    # if csv not uploaded, redirect to csvupload 
    if 'dataLoad' not in session:
        return render_template("csvupload.html")

    # if feature selection not done, redirect to featurechecker
    if not session['dataLoad']:
        return redirect(url_for('featurechecker'))

    else:
        return redirect(url_for('dashboard'))

@app.route('/csvuploader', methods=['GET', 'POST'])
def csvuploader():
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    if request.method == 'POST':
        # check if a file is submitted or not
        if 'csv-file' not in request.files:
            flash('No file part')
            return redirect(url_for('datasetSubmit'))

        file = request.files['csv-file']

        # check if file name is empty
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('datasetSubmit'))

        # if file uploaded is csv
        ## read csv into a dataframe
        ## extract and safe dataframe columns
        if file and uploadChecker(file.filename):
            dataframe = pd.read_csv(file)
            dataframe = dataframe.fillna(dataframe.mean())
            # dataframe.dropna(inplace = True)
            session['dataframe'] = dataframe.to_dict('list')

            dataColumns = list(dataframe.columns)
            print(dataColumns)
            session['dataColumns'] = dataColumns
            print(session['dataColumns'])

        dataLoad = False
        session['dataLoad'] = dataLoad

        return redirect(url_for('featurechecker'))

@app.route('/featurechecker', methods=['GET', 'POST'])
def featurechecker():
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    checkboxes = ''''''
    for column in dataColumns:
        checkboxes += f'''
            <tr>
                <td>{ column }</td>
                <td><input type = "checkbox" name = "{ column + "-is" }" value = "off"/></td>
            </tr>
        '''

    # print(Markup(checkboxes))
    dataLoad = True
    session['dataLoad'] = dataLoad

    return render_template("featurechecker.html", checkboxes = Markup(checkboxes))
        
# gets checkbox input from feature selection list, sets columns we wanna use
@app.route('/updateFeatureList', methods=['GET', 'POST'])
def updateFeatureList():
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    # extract wanted features
    tcol = []
    for col in dataColumns:
        if request.form.getlist(f'{col}-is'):
            tcol.append(col)
        else:
            pass

    # assign columns
    dataColumns = tcol
    session['dataColumns'] = dataColumns
    print(tcol)

    # making numerical df
    dataframe_num = dataframe.copy()
    for col in dataframe_num.columns:
        if dataframe_num[col].dtype == np.dtype('O'):
            dataframe_num[col] = dataframe_num[col].astype('category')
            dataframe_num[col] = dataframe_num[col].cat.codes
    session['dataframe_num'] = dataframe_num

    return redirect(url_for('dashboard'))

# route for dashboard
@app.route('/index')
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    # if csv not uploaded, redirect to csvupload 
    if 'dataLoad' not in session:
        return render_template("csvupload.html")

    # if feature selection not done, redirect to featurechecker
    if not session['dataLoad']:
        return redirect(url_for('featurechecker'))

    # get dataframe information
    tableInfo = f''

    # get column information
    columnInfo = ''''''
    for i, col in enumerate(dataColumns):

        t = "<br>".join(re.split(r'(?:\r\n|\r|\n)', escape(dataframe[col].describe())))
        columnInfo += f'''
            <div class="card shadow mb-4">
                <!-- Card Header - Accordion -->
                <a href="#collapseCardExample{i}" class="d-block card-header py-3" data-toggle="collapse"
                    role="button" aria-expanded="true" aria-controls="collapseCardExample{i}">
                    <h6 class="m-0 font-weight-bold text-primary">{ col } Information</h6>
                </a>
                <!-- Card Content - Collapse -->
                <div class="collapse show" id="collapseCardExample{i}">
                    <div class="card-body">
                        { t }
                    </div>
                </div>
            </div>
        '''

    # session['dataframe'] = dataframe.to_dict('list')
    # session['dataLoad'] = dataLoad
    # session['dataColumns'] = dataColumns
    # session['dataframe_num'] = dataframe_num
    return render_template("dashboard.html", tableInfo = Markup(tableInfo), columnInfo = Markup(columnInfo))

# route to display dataframe/table
@app.route('/tables')
def tables():
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    # if csv not uploaded, redirect to csvupload 
    if 'dataLoad' not in session:
        return render_template("csvupload.html")

    # if feature selection not done, redirect to featurechecker
    if not session['dataLoad']:
        return redirect(url_for('featurechecker'))

    # display table header and footer
    tcol = ''
    for col in dataColumns:
        tcol += f'''
            <th>{ col }</th>
        '''
    headfoot = f'''
        <thead>
            { tcol }
        </thead>
        <tfoot>
            { tcol }
        </tfoot>
    '''

    # display top 100 rows
    tablebody = ''''''
    for i in range(min(100, len(dataframe.index))):
        trow = '''
            <tr>
        '''
        for j in dataColumns:
            trow += f'''
                <td>{ dataframe[j].iloc[i] }</td>
            '''
        trow += '''
            </tr>
        '''
        tablebody += trow

    # session['dataframe'] = dataframe.to_dict('list')
    # session['dataLoad'] = dataLoad
    # session['dataColumns'] = dataColumns
    # session['dataframe_num'] = dataframe_num
    return render_template("tables.html", headfoot = Markup(headfoot), tablebody = Markup(tablebody))

@app.route('/charts')
def charts():
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    # if csv not uploaded, redirect to csvupload 
    if 'dataLoad' not in session:
        return render_template("csvupload.html")

    # if feature selection not done, redirect to featurechecker
    if not session['dataLoad']:
        return redirect(url_for('featurechecker'))

    # generate plots
    distb64 = getDistplot()
    countb64 = getCountplot()
    pairb64 = getPairplot()
    candleb64 = getCandleplot()

    # session['dataframe'] = dataframe.to_dict('list')
    # session['dataLoad'] = dataLoad
    # session['dataColumns'] = dataColumns
    # session['dataframe_num'] = dataframe_num
    return render_template("charts.html", distb64 = distb64, countb64 = countb64, pairb64 = pairb64, candleb64 = candleb64)

@app.route('/test')
def test():
    return render_template("dashboard.html")

@app.route('/reupload')
def reupload():
    # reset session variables 
    # session.pop('', None)
    session.pop('dataframe', None)
    session.pop('dataLoad', None)
    session.pop('dataColumns', None)
    session.pop('dataframe_num', None)
    session.pop('mlType', None)
    session.pop('trainX', None)
    session.pop('trainY', None)
    session.pop('targetVar', None)

    return redirect(url_for('datasetSubmit'))

@app.route('/selectTarget', methods=['GET', 'POST'])
def selectTarget():
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    checkboxes = ''''''
    for column in dataColumns:
        checkboxes += f'''
            <tr>
                <td>{ column }</td>
                <td><input type = "radio" name = "target" value = "{ column }"/></td>
            </tr>
        '''

    return render_template("selectTarget.html", checkboxes = Markup(checkboxes))

@app.route('/setTarget', methods=['GET', 'POST'])
def setTarget():
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None

    # get and store target variable
    session['targetVar'] = request.form.getlist(f'target')[0]

    # pre-process dataframes
    ## generate train Features
    trainX = dataframe.loc[:, [col for col in dataColumns if col != session['targetVar']]]

    ## get categorical features and one hot encode
    categoricalFeatures = [col for col in list(trainX.columns) if trainX[col].dtype == 'object']
    for col in categoricalFeatures:
        # temporariry story categorical col and delete from trainDF
        t = trainX[col]
        trainX.drop([col], axis = 1, inplace = True)

        # generate one-hot
        oneHot = pd.get_dummies(t, prefix=col)

        # add one-hot to trainDF
        trainX = pd.concat([trainX, oneHot], axis = 1)
    
    ## generate train Target
    trainY = dataframe.loc[:, session['targetVar']]
    standardScaler = StandardScaler()
    session['trainX'] = standardScaler.fit_transform(trainX.values)

    if dataframe[session['targetVar']].dtype == 'object':
        # if the problem is of classification we would have to one-hot target too
        trainY = pd.get_dummies(trainY)
        session['trainY'] = trainY.values
        session['mlType'] = 'Classification'

    else:
        session['mlType'] = 'Regression'
        session['trainY'] = trainY.values

    print(len(session['trainX']))
    print(len(session['trainY']))

    # can present models we want to test
    return render_template("disptarget.html", algoName = Markup(session['mlType']))
    
@app.route('/predicitveAnalysis', methods=['GET', 'POST'])
def predicitveAnalysis():
    if 'mlType' not in session:
        return redirect(url_for('selectTarget'))
    
    mlType = session['mlType']
    trainX = session['trainX']
    trainY = session['trainY']

    if mlType == 'Regression':
        modelName = ['Linear Regression', 'Ridge Regression', 'Support Vector Regression']
        models = [LinearRegression(), Ridge(), SVR()]
    elif mlType == 'Classification':
        modelName = ['Logistic Classification', 'Random Forest Classification', 'Support Vector Classification']
        models = [LogisticRegression(), RandomForestClassifier(), SVC()]
    else:
        return redirect(url_for('selectTarget'))

    crossValidationAccPlot = {}
    crossValidationAccAvg = {}
    for idx, model in enumerate(models):
        kFold = 10
        tList = cross_val_score(model, trainX, trainY, cv = kFold)
        crossValidationAccPlot[modelName[idx]] = getAccplot(tList, kFold, modelName[idx])
        crossValidationAccAvg[modelName[idx]] = np.mean(tList)

    html = ''''''
    for idx, key in enumerate(modelName):
        html += f'''
            <div class="card shadow mb-4">
                <a href="#collapseCardExample{ idx }" class="d-block card-header py-3" data-toggle="collapse"
                    role="button" aria-expanded="true" aria-controls="collapseCardExample{ idx }">
                    <h6 class="m-0 font-weight-bold text-primary">{ key } - Mean Accuracy { round(crossValidationAccAvg[key], 2) }</h6>
                </a>

                <div class="collapse show" id="collapseCardExample{ idx }">
                    <div class="card-body">
                        <img src = "data:image/png;base64, { crossValidationAccPlot[key] }" style='height: 100%; width: 100%; object-fit: contain'>
                    </div>
                </div>
            </div>
        '''
        

    return render_template("pred.html", htmlCode = Markup(html))


########################################################################################################### Helper Functions

def uploadChecker(filename):
    '''
    Returns wether file can be accepted or not based on it's name.

    Arguments:
        filename - Name of the file
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getDistplot():
    '''
    Plots distribution plot for all dataframe columns and saves the plot.

    Arguments:
        path - Location where to save the plot 
    '''
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    fig, ax = plt.subplots(len(dataColumns), figsize=(len(dataColumns) * 2, len(dataColumns) * 2))

    for i, col in enumerate(dataColumns):
        sns.distplot(dataframe_num[col], hist=True, ax=ax[i])
        ax[i].set_ylabel(col, fontsize=8)

    img = BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)

    img.seek(0)
    b64img = base64.b64encode(img.getvalue()).decode('utf8')

    return b64img

def getCountplot():
    '''
    Plots count/histogram plot for all categorical dataframe columns and saves the plot.
    '''
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    tcols = []
    for col in dataColumns:
        if len(dataframe[col].unique()) < 20:
            tcols.append(col)

    i = 0
    fig, ax = plt.subplots(1, len(tcols), figsize=(len(tcols) * 2, len(tcols) * 2))
    for i, col in enumerate(tcols):
        dataframe[col].value_counts().plot.bar(ax=ax[i])
        ax[i].set_title(col, fontsize=10)

    img = BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)

    img.seek(0)
    b64img = base64.b64encode(img.getvalue()).decode('utf8')

    return b64img

def getPairplot():
    '''
    Plots pair plot for all dataframe column and saves the plot.
    '''
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    tdf = dataframe_num[dataColumns]
    # pp = sns.pairplot(tdf)

    f, ax = plt.subplots(figsize=(11, 9))
    corr = tdf.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    img = BytesIO()
    f.savefig(img, format='png')

    img.seek(0)
    b64img = base64.b64encode(img.getvalue()).decode('utf8')

    return b64img

def getCandleplot():
    '''
    Plots candle plot for all dataframe column and saves the plot.
    '''
    dataframe = pd.DataFrame(session['dataframe']) if 'dataframe' in session else None
    dataLoad = session['dataLoad'] if 'dataLoad' in session else None
    dataColumns = session['dataColumns'] if 'dataColumns' in session else None
    dataframe_num = session['dataframe_num'] if 'dataframe_num' in session else None

    tcols = []
    for col in dataColumns:
        if dataframe[col].dtype != np.dtype('O'):
            tcols.append(col)
            
    fig, ax = plt.subplots(len(tcols), figsize=(len(tcols) * 2, len(tcols) * 4))

    for i, col in enumerate(tcols):
        sns.boxplot(y = dataframe[col], ax = ax[i])
        ax[i].set_xlabel(col, fontsize=8)

    img = BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)

    img.seek(0)
    b64img = base64.b64encode(img.getvalue()).decode('utf8')

    return b64img

def getAccplot(accList, kFold, modelName):
    fig, ax = plt.subplots(1, figsize=(5, 5))

    plt.plot(range(1, kFold + 1), accList, 'g', label = modelName)
    plt.title('Cross-Validation accuracy')
    plt.xlabel('K-Fold')
    plt.ylabel('Accuracy')
    plt.legend()

    img = BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)

    img.seek(0)
    b64img = base64.b64encode(img.getvalue()).decode('utf8')

    return b64img
    


########################################################################################################### Main

if __name__ == "__main__":
    app.run()