

from flask import Flask, render_template, jsonify
from main import run_analysis

app = Flask(__name__, template_folder='../frontend/templates')


# Set '/' and '/login' to both render the login page
@app.route('/')
@app.route('/login')
def home():
    return render_template('login.html')

@app.route('/search')
def search():
    return render_template('search and wishlist.html')


# New route to return analysis results as JSON
@app.route('/analyze')
def analyze():
    results = run_analysis()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)