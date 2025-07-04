from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and residual std
try:
    model = joblib.load(open('gradient_model.pkl', 'rb'))
    residual_std = joblib.load(open('gradient_residual_std.pkl', 'rb'))
except Exception as e:
    model = None
    residual_std = None
    print(f"Model loading failed: {e}")

# Dropdown options
product_list = sorted([
    'Lamb', 'Beef', 'Oranges', 'Milk', 'Bananas', 'Potatoes', 'Pork',
    'Rice', 'Butter', 'Peppers', 'Apples', 'Rye', 'Cheese',
    'Blueberries', 'Oats', 'Corn', 'Grapes', 'Wheat', 'Chicken',
    'Tomatoes', 'Onions', 'Barley', 'Yogurt', 'Cabbage', 'Peaches',
    'Lettuce', 'Strawberries', 'Carrots'
])

months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']


@app.route('/')
def home():
    return render_template('index.html', products=product_list, months=months, form_data={})


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not residual_std:
        return render_template('index.html',
                               products=product_list,
                               months=months,
                               prediction="Model not available. Please check deployment.",
                               form_data=request.form)
    try:
        # Fallback logic
        product = request.form['product'] or request.form['product_select']
        month = request.form['month'] or request.form['month_select']
        units_shipped = request.form['units_shipped_kg'] or request.form['units_shipped_select']
        units_on_hand = request.form['units_on_hand_kg'] or request.form['units_on_hand_select']
        price_per_kg = float(request.form['price_per_kg'])

        data = {
            'product_name': product,
            'units_shipped_kg': float(units_shipped),
            'units_on_hand_kg': float(units_on_hand),
            'price_per_kg': price_per_kg,
            'month': month
        }

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        lower = max(prediction - residual_std, 0)
        upper = prediction + residual_std

        prediction_text = (
            f"Predicted Units Sold for <b>{product}</b>: <strong>{prediction:.2f} kg</strong><br>"
            f"Confidence Interval: <span style='color:darkblue;'>{lower:.2f} – {upper:.2f} kg</span> (± {residual_std:.0f})"
        )

        return render_template('index.html',
                               products=product_list,
                               months=months,
                               prediction=prediction_text,
                               form_data=request.form)
    except Exception as e:
        return render_template('index.html',
                               products=product_list,
                               months=months,
                               prediction=f"Error: {str(e)}",
                               form_data=request.form)


if __name__ == '__main__':
    app.run(debug=True)
