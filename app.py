from flask import Flask, render_template, flash, redirect, request, url_for
from werkzeug.utils import secure_filename
from markupsafe import Markup, escape
import re
import os

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

########################################################################################################### Variables
### Flask Variables
UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'csv'}

### Variables used throughout our code
dataframe = None
dataLoad = None
dataColumns = None
dataframe_num = None


########################################################################################################### Flask routes
# route for csv upload page
@app.route('/')
@app.route('/index')
def datasetSubmit():
    # if csv not uploaded, redirect to csvupload 
    if dataLoad == None:
        return render_template("csvupload.html")

    # if feature selection not done, redirect to featurechecker
    if not dataLoad:
        return redirect(url_for('featurechecker'))

    else:
        return redirect(url_for('dashboard'))

@app.route('/csvuploader', methods=['GET', 'POST'])
def csvuploader():
    global dataframe
    global dataColumns
    global dataLoad

    # delete past plots
    deletePlots()

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
            dataColumns = list(dataframe.columns)

        dataLoad = False
        return redirect(url_for('featurechecker'))

@app.route('/featurechecker', methods=['GET', 'POST'])
def featurechecker():
    global dataframe
    global dataColumns
    global dataLoad

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
    return render_template("featurechecker.html", checkboxes = Markup(checkboxes))
        
# gets checkbox input from feature selection list, sets columns we wanna use
@app.route('/updateFeatureList', methods=['GET', 'POST'])
def updateFeatureList():
    global dataColumns
    global dataframe_num

    # extract wanted features
    tcol = []
    for col in dataColumns:
        if request.form.getlist(f'{col}-is'):
            tcol.append(col)
        else:
            pass

    # assign columns
    dataColumns = tcol
    print(tcol)

    # making numerical df
    dataframe_num = dataframe.copy()
    for col in dataframe_num.columns:
        if dataframe_num[col].dtype == np.dtype('O'):
            dataframe_num[col] = dataframe_num[col].astype('category')
            dataframe_num[col] = dataframe_num[col].cat.codes

    return redirect(url_for('dashboard'))

# route for dashboard
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    global dataframe
    global dataColumns
    global dataLoad

    # if csv not uploaded, redirect to csvupload 
    if dataLoad == None:
        return render_template("csvupload.html")

    # if feature selection not done, redirect to featurechecker
    if not dataLoad:
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

    return render_template("dashboard.html", tableInfo = Markup(tableInfo), columnInfo = Markup(columnInfo))

# route to display dataframe/table
@app.route('/tables')
def tables():
    # if csv not uploaded, redirect to csvupload 
    if dataLoad == None:
        return render_template("csvupload.html")

    # if feature selection not done, redirect to featurechecker
    if not dataLoad:
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

    return render_template("tables.html", headfoot = Markup(headfoot), tablebody = Markup(tablebody))

@app.route('/charts')
def charts():
    # if csv not uploaded, redirect to csvupload 
    if dataLoad == None:
        return render_template("csvupload.html")

    # if feature selection not done, redirect to featurechecker
    if not dataLoad:
        return redirect(url_for('featurechecker'))

    # generate plots if they dont exists
    if not os.path.exists("./static/plots/distplot.jpg"):
        getDistplot()

    if not os.path.exists("./static/plots/countplot.jpg"):
        getCountplot()

    if not os.path.exists("./static/plots/pairplot.jpg"):
        getPairplot()

    if not os.path.exists("./static/plots/candleplot.jpg"):
        getCandleplot()

    return render_template("charts.html")

@app.route('/test')
def test():
    return render_template("dashboard.html")

@app.route('/reupload')
def reupload():
    global dataframe
    global dataLoad
    global dataColumns
    global dataframe_num

    dataframe = None
    dataLoad = None
    dataColumns = None
    dataframe_num = None

    return redirect(url_for('datasetSubmit'))

########################################################################################################### Helper Functions

def uploadChecker(filename):
    '''
    Returns wether file can be accepted or not based on it's name.

    Arguments:
        filename - Name of the file
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getDistplot(path = './static/plots/distplot.jpg'):
    '''
    Plots distribution plot for all dataframe columns and saves the plot.

    Arguments:
        path - Location where to save the plot 
    '''
    fig, ax = plt.subplots(len(dataColumns), figsize=(len(dataColumns) * 2, len(dataColumns) * 2))

    for i, col in enumerate(dataColumns):
        sns.distplot(dataframe_num[col], hist=True, ax=ax[i])
        ax[i].set_ylabel(col, fontsize=8)

    fig.savefig(path)
    plt.close(fig)

def getCountplot(path = './static/plots/countplot.jpg'):
    '''
    Plots count/histogram plot for all categorical dataframe columns and saves the plot.

    Arguments:
        path - Location where to save the plot 
    '''
    tcols = []
    for col in dataColumns:
        if len(dataframe[col].unique()) < 20:
            tcols.append(col)

    i = 0
    fig, ax = plt.subplots(1, len(tcols), figsize=(len(tcols) * 2, len(tcols) * 2))
    for i, col in enumerate(tcols):
        dataframe[col].value_counts().plot.bar(ax=ax[i])
        ax[i].set_title(col, fontsize=10)

    fig.savefig(path)
    plt.close(fig)

def getPairplot(path = './static/plots/pairplot.jpg'):
    '''
    Plots pair plot for all dataframe column and saves the plot.

    Arguments:
        path - Location where to save the plot 
    '''
    tdf = dataframe_num[dataColumns]
    pp = sns.pairplot(tdf)
    pp.savefig(path)

def getCandleplot(path = './static/plots/candleplot.jpg'):
    '''
    Plots candle plot for all dataframe column and saves the plot.

    Arguments:
        path - Location where to save the plot 
    '''
    tcols = []
    for col in dataColumns:
        if dataframe[col].dtype != np.dtype('O'):
            tcols.append(col)
            
    fig, ax = plt.subplots(len(tcols), figsize=(len(tcols) * 2, len(tcols) * 4))

    for i, col in enumerate(tcols):
        sns.boxplot(y = dataframe[col], ax = ax[i])
        ax[i].set_xlabel(col, fontsize=8)

    fig.savefig(path)
    plt.close(fig)

def deletePlots():
    '''
    Deletes saved plots.
    '''
    try:
        os.remove('./static/plots/candleplot.jpg')
        os.remove('./static/plots/pairplot.jpg')
        os.remove('./static/plots/countplot.jpg')
        os.remove('./static/plots/distplot.jpg')
    except:
        pass

########################################################################################################### Main

if __name__ == "__main__":
    app.run()