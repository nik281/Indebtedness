from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

app = Flask(__name__)

model = joblib.load('random_forest_model.joblib')
df = pd.read_csv('debt-investment-and-asset-analysis.csv', index_col='id')

# Encode state_code
le = LabelEncoder()
df['state_code'] = le.fit_transform(df['state_code'])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        debt_asset_ratio = float(request.form['debt_asset_ratio'])
        avg_amount_of_debt = float(request.form['avg_amount_of_debt'])
        state_code = int(request.form['state_code'])
        avg_value_assets = float(request.form['avg_value_assets'])
        fxd_cap_exp = float(request.form['fxd_cap_exp'])
        hh_reporting_fxd_cap_exp = float(request.form['hh_reporting_fxd_cap_exp'])

        input_data = pd.DataFrame({
            'debt_asset_ratio': [debt_asset_ratio],
            'avg_amount_of_debt': [avg_amount_of_debt],
            'state_code': [state_code],
            'avg_value_assets': [avg_value_assets],
            'fxd_cap_exp': [fxd_cap_exp],
            'hh_reporting_fxd_cap_exp': [hh_reporting_fxd_cap_exp]
        })

        prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

@app.route('/visualizations')
def visualizations():
    # Create interactive visualizations using Plotly
    
    # Scatter plot
    scatter = px.scatter(df, x='avg_amount_of_debt', y='avg_value_assets', 
                         title='Debt vs Assets')
    
    # Histogram
    histogram = px.histogram(df, x='debt_asset_ratio', 
                             title='Distribution of Debt-Asset Ratio')
    
    # Box plot
    boxplot = px.box(df, x='state_code', y='fxd_cap_exp', 
                     title='Fixed Capital Expenditure by State')
    
    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    heatmap = px.imshow(corr_matrix, title='Correlation Heatmap')
    
    # Combine all plots into a single figure
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Debt vs Assets', 
                                                        'Distribution of Debt-Asset Ratio',
                                                        'Fixed Capital Expenditure by State',
                                                        'Correlation Heatmap'),vertical_spacing=0.1, horizontal_spacing=0.2)
    
    fig.add_trace(scatter.data[0], row=1, col=1)
    fig.add_trace(histogram.data[0], row=1, col=2)
    fig.add_trace(boxplot.data[0], row=2, col=1)
    fig.add_trace(heatmap.data[0], row=2, col=2)
    
    fig.update_layout(height=1000, width=1200, title_text="Data Visualizations")
    
    # Convert the figure to JSON for rendering in the template
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Generate summary statistics
    stats = df.describe().to_html(classes='table table-striped')
    
    return render_template('visualizations.html', plot_json=plot_json, stats=stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')