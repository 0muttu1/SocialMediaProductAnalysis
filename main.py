from flask import Flask, request, redirect, url_for, render_template, flash
import final  # import your final module
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
secret_key = os.urandom(24)

app = Flask(__name__) # replace with your secret key

app.secret_key = secret_key
server_busy = false

@app.route("/", methods=['GET', 'POST'])
def HOME():
    return redirect('productanalysis')

@app.route("/productanalysis", methods=['GET', 'POST'])
def product_analysis():
    if request.method == 'POST':
        query = request.form.get('query')
        nos = int(request.form.get('nos'))
        if query and nos:  # check if 'query' and 'nos' are not None
            
            result = final.main(query, nos)  # call the main function of your final script with parameters
            flash(result)  # store the result in a flash message
            return redirect(url_for('results', query=query))
    return render_template('product_analysis_form.html')  # render enter_data.html template for GET requests

@app.route("/results/<query>")
def results(query):
    # Read the CSV file
    df = pd.read_csv(f'Results/{query}sentimentdata.csv')

    # Count the number of each sentiment
    sentiment_counts = df['Sentiment'].value_counts()

    # Read the topics.txt file
    with open(f'Results/{query}Topics.txt', 'r') as file:
        topics = file.read().splitlines()

    # Prepare the data for JSON
    data = {
        'sentiment_counts': sentiment_counts.to_dict(),
        'topics': topics
    }

    # Convert the data to JSON
    json_data = json.dumps(data)

    with open(f'Results/{query}_data.json', 'w') as file:
        file.write(json_data)
    # Render the results.html template with the result and json_data
    plot_url = url_for('static', filename=f'Results/{query}_plot.png')
    print(plot_url)
    return render_template('results.html', data=data,plot_url=plot_url)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

# if __name__ ==  "__main__":
#     app.run()
