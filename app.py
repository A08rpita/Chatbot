from flask import Flask, request, jsonify, send_file
import requests
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    data = request.get_json()
    try:
        source_currency = data['queryResult']['parameters']['unit-currency']['currency']
        amount = data['queryResult']['parameters']['unit-currency']['amount']
        target_currency = data['queryResult']['parameters']['currency-name']
    except KeyError:
        return jsonify({
            'fulfillmentText': "Invalid input. Please provide valid currencies and amount."
        })
    
    cf = fetch_conversion_factor(source_currency, target_currency)
    if cf is None:
        return jsonify({
            'fulfillmentText': "Sorry, I couldn't fetch the conversion rate at this time."
        })
    
    final_amount = amount * cf
    final_amount = round(final_amount, 2)

    # Generate the graph and save it as an image
    graph_path = generate_graph(source_currency, target_currency)

    response = {
        'fulfillmentText': "{} {} is {} {}. Here is the historical trend:".format(
            amount, source_currency, final_amount, target_currency
        )
    }
    return jsonify(response), send_file(graph_path, mimetype='image/png')


def fetch_conversion_factor(source, target):
    api_key = "9aa0c54f5ad4c460c36d"  # Replace with your environment variable in production
    url = f"https://free.currconv.com/api/v7/convert?q={source}_{target}&compact=ultra&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data[f"{source}_{target}"]
    except (requests.exceptions.RequestException, KeyError):
        return None


def fetch_historical_data(source, target):
    api_key = "9aa0c54f5ad4c460c36d"  # Replace with your environment variable in production
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Fetch data for the last 7 days
    url = (f"https://free.currconv.com/api/v7/convert?q={source}_{target}&compact=ultra"
           f"&date={start_date.strftime('%Y-%m-%d')}&endDate={end_date.strftime('%Y-%m-%d')}&apiKey={api_key}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        key = f"{source}_{target}"
        return data.get(key, {})
    except (requests.exceptions.RequestException, KeyError):
        return {}


def generate_graph(source, target):
    historical_data = fetch_historical_data(source, target)
    if not historical_data:
        return "No data available to generate graph."

    dates = list(historical_data.keys())
    rates = list(historical_data.values())

    # Plot the historical data
    plt.figure(figsize=(10, 6))
    plt.plot(dates, rates, marker='o', linestyle='-', color='blue')
    plt.title(f"Exchange Rate Trend: {source} to {target}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"Rate ({target} per {source})", fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the graph to a BytesIO object
    graph = io.BytesIO()
    plt.savefig(graph, format='png')
    graph.seek(0)
    plt.close()

    # Save the graph locally as a fallback
    graph_path = f"{source}_{target}_trend.png"
    with open(graph_path, "wb") as f:
        f.write(graph.read())
    graph.seek(0)  # Reset pointer for sending via Flask
    return graph


if __name__ == "__main__":
    app.run(debug=True)
