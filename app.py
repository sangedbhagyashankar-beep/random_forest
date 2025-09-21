import os
import json
from datetime import datetime
from flask import Flask, render_template, request, session, flash, redirect, jsonify, url_for, g
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# --- App Configuration ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    features_json = db.Column(db.String(500))
    prediction_result = db.Column(db.String(50))
    confidence = db.Column(db.Float)


# --- Load Model and Data ---
def get_data():
    if 'df' not in g:
        try:
            g.df = pd.read_csv('heart.csv')
            g.feat_imp_df = pd.read_csv('feature_importances.csv')
            g.model = joblib.load('heart_model.pkl')
            g.feature_cols = g.df.drop('target', axis=1).columns.tolist()
        except FileNotFoundError:
            g.df = None
            g.feat_imp_df = None
            g.model = None
            g.feature_cols = None
            print("Warning: model or data files not found. Please run train_model.py first.")
    return g.df, g.feat_imp_df, g.model, g.feature_cols

@app.before_request
def load_data():
    get_data()

# --- Routes ---
@app.route("/")
def home():
    if not g.model:
        flash("Model not found. Please train the model by running train_model.py.", "danger")
        return render_template("home.html", feature_cols=[])
    return render_template("home.html", feature_cols=g.feature_cols)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            session["username"] = user.username
            flash("Logged in successfully!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists. Please choose a different one.", "danger")
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        flash("Please log in to make a prediction.", "warning")
        return redirect(url_for("login"))
    if not g.model:
        flash("Model not found.", "danger")
        return redirect(url_for("home"))
        
    try:
        features = {feat: float(request.form[feat]) for feat in g.feature_cols}
        input_df = pd.DataFrame([features])
        
        prediction_prob = g.model.predict_proba(input_df)[0]
        prediction = g.model.predict(input_df)[0]
        
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
        confidence = round(prediction_prob[prediction] * 100, 2)

        # Save prediction history and get the ID
        new_prediction = Prediction(
            user_id=session["user_id"],
            features_json=json.dumps(features),
            prediction_result=result,
            confidence=confidence
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        # Redirect to the new report page
        return redirect(url_for("report", prediction_id=new_prediction.id))
        
    except (ValueError, KeyError) as e:
        flash(f"Invalid input: {e}", "danger")
        return redirect(url_for("home"))

@app.route("/report/<int:prediction_id>")
def report(prediction_id):
    if "user_id" not in session:
        flash("Please log in to view the report.", "warning")
        return redirect(url_for("login"))

    prediction = Prediction.query.filter_by(id=prediction_id, user_id=session["user_id"]).first()
    if not prediction:
        flash("Report not found or you do not have permission to view it.", "danger")
        return redirect(url_for("home"))

    features = json.loads(prediction.features_json)
    
    # 1. Prediction Probabilities Plot
    input_df = pd.DataFrame([features])
    prediction_prob = g.model.predict_proba(input_df)[0]
    plot_data = go.Bar(x=["No Heart Disease", "Heart Disease"], y=prediction_prob,
                        marker_color=['#42a5f5', '#ef5350'])
    plot_layout = go.Layout(title="Prediction Probabilities", yaxis_title="Probability")
    prob_fig = go.Figure(data=[plot_data], layout=plot_layout)
    prediction_plot_html = pio.to_html(prob_fig, full_html=False)

    # 2. Feature Importance Plot
    feat_imp_df_sorted = g.feat_imp_df.sort_values(by='Importance', ascending=False)
    feat_imp_fig = px.bar(feat_imp_df_sorted.head(5), x='Feature', y='Importance',
                          title="Top 5 Most Influential Features",
                          labels={"Feature": "Model Feature", "Importance": "Relative Importance"},
                          color="Importance", color_continuous_scale=px.colors.sequential.Plasma)
    feature_importance_html = pio.to_html(feat_imp_fig, full_html=False)

    # 3. Analyze most responsible features
    responsible_features = {}
    for index, row in feat_imp_df_sorted.iterrows():
        feat = row['Feature']
        # Compare user's value to the median of the entire dataset
        median_val = g.df[feat].median()
        user_val = features[feat]
        
        # Create insight based on comparison
        if user_val > median_val:
            responsible_features[feat] = {
                'value': user_val,
                'status': 'high',
                'insight': f"Your {feat.replace('_', ' ').title()} value is higher than the average, which is a significant factor in your prediction."
            }
        elif user_val < median_val:
            responsible_features[feat] = {
                'value': user_val,
                'status': 'low',
                'insight': f"Your {feat.replace('_', ' ').title()} value is lower than the average, which is a significant factor in your prediction."
            }
        else:
            responsible_features[feat] = {
                'value': user_val,
                'status': 'normal',
                'insight': f"Your {feat.replace('_', ' ').title()} value is close to the average, indicating a moderate influence on your prediction."
            }
        # Only show the top 3 most important features
        if len(responsible_features) >= 3:
            break

    # 4. Determine "Severity"
    severity = "Low"
    if prediction.confidence >= 80 and "Heart Disease" in prediction.prediction_result:
        severity = "High"
    elif prediction.confidence >= 50 and "Heart Disease" in prediction.prediction_result:
        severity = "Moderate"
    elif prediction.confidence < 50 and "Heart Disease" in prediction.prediction_result:
        severity = "Low (unlikely)"

    return render_template(
        "report.html",
        prediction=prediction,
        features=features,
        prediction_plot_html=prediction_plot_html,
        feature_importance_html=feature_importance_html,
        responsible_features=responsible_features,
        severity=severity
    )

@app.route("/history")
def history():
    if "user_id" not in session:
        flash("Please log in to view your history.", "warning")
        return redirect(url_for("login"))
    user = User.query.get(session["user_id"])
    predictions = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template("history.html", predictions=predictions)

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please log in to view the dashboard.", "warning")
        return redirect(url_for("login"))
    
    df, feat_imp_df, _, _ = get_data()
    if df is None or feat_imp_df is None:
        flash("Required data files not found. Please make sure heart.csv and feature_importances.csv exist.", "danger")
        return redirect(url_for("home"))
    
    # 1. Heart Disease Distribution (Pie Chart Data)
    target_counts = df['target'].value_counts()
    pie_labels = ['No Heart Disease', 'Heart Disease']
    pie_values = [int(target_counts.get(0, 0)), int(target_counts.get(1, 0))]
    
    # 2. Heart Disease Count by Age Group (Bar Chart Data)
    bins = [20, 30, 40, 50, 60, 70, 80, 90]
    labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    heart_disease_by_age = df[df['target'] == 1]['age_group'].value_counts().sort_index()
    bar_labels = heart_disease_by_age.index.tolist()
    bar_data = heart_disease_by_age.values.tolist()
    
    # 3. Model Feature Importances (Bar Chart Data)
    feat_imp_df_sorted = feat_imp_df.sort_values(by='Importance', ascending=False)
    feat_imp_labels = feat_imp_df_sorted['Feature'].tolist()
    feat_imp_data = feat_imp_df_sorted['Importance'].tolist()
    
    return render_template("dashboard.html",
                           pie_labels=pie_labels,
                           pie_values=pie_values,
                           bar_labels=bar_labels,
                           bar_data=bar_data,
                           feat_imp_labels=feat_imp_labels,
                           feat_imp_data=feat_imp_data)

@app.route("/distributions")
def distributions():
    if "user_id" not in session:
        flash("Please log in to view the distributions.", "warning")
        return redirect(url_for("login"))
    df, _, _, _ = get_data()
    if df is None:
        flash("Required data file 'heart.csv' not found. Please make sure the file exists.", "danger")
        return redirect(url_for("home"))

    key_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    histogram_htmls = []
    
    for feature in key_features:
        fig = px.histogram(df, x=feature, color='target',
                           title=f'Distribution of {feature.capitalize()} by Target',
                           barmode='overlay',
                           color_discrete_map={0: 'blue', 1: 'red'},
                           labels={'target': 'Heart Disease'})
        histogram_htmls.append(pio.to_html(fig, full_html=False))

    return render_template("distributions.html", histogram_htmls=histogram_htmls)

@app.route("/plotly_plot")
def plotly_plot():
    if "user_id" not in session:
        flash("Please log in to view the interactive plot.", "warning")
        return redirect(url_for("login"))
    _, feat_imp_df, _, _ = get_data()
    if feat_imp_df is None:
        flash("Required data file 'feature_importances.csv' not found. Please make sure the file exists.", "danger")
        return redirect(url_for("home"))

    fig = px.bar(feat_imp_df, x='Feature', y='Importance',
                 title="Interactive Feature Importances for Heart Disease Prediction",
                 labels={"Feature": "Model Feature", "Importance": "Relative Importance"},
                 color="Importance", color_continuous_scale=px.colors.sequential.Plasma)
    
    graph_html = pio.to_html(fig, full_html=False)

    return render_template("plotly_plot.html", plot_html=graph_html)
    
@app.route("/heatmap_plot")
def heatmap_plot():
    if "user_id" not in session:
        flash("Please log in to view the heatmap.", "warning")
        return redirect(url_for("login"))
    df, _, _, _ = get_data()
    if df is None:
        flash("Required data file 'heart.csv' not found. Please make sure the file exists.", "danger")
        return redirect(url_for("home"))

    corr = df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='Viridis'
    ))
    fig.update_layout(
        title='Feature Correlation Heatmap',
        xaxis_nticks=36,
        yaxis_autorange='reversed'
    )
    heatmap_html = pio.to_html(fig, full_html=False)
    
    return render_template('heatmap.html', heatmap_html=heatmap_html)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)