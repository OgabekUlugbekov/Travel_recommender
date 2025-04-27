import logging
logging.basicConfig(
    filename='travel_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'  # -> to track every important step of this progam
)
import pyodbc
class DatabaseManager:
    def __init__(self):
        try:
            self.conn = pyodbc.connect(
                'DRIVER={SQL Server};'
                'SERVER=DESKTOP-XXXX\\SQLEXPRESS;'
                'DATABASE=TravelDB;'
                'Trusted_Connection=yes;'
            )  # -> connect to sql server database
            self.cursor = self.conn.cursor()
            logging.info("Databse conected sucessfully")  # -> write in diary we connected
        except Exception as e:
            logging.error(f"Failed to conect to databse: {e}")  # -> write error if we cant connect
            raise

    def fetch_data(self, query: str, params: tuple = ()) -> list:
        try:
            self.cursor.execute(query, params)  # -> run sql query to get data
            rows = self.cursor.fetchall()
            logging.info("Got data from databse")
            return rows
        except Exception as e:
            logging.error(f"Failed to fetch data: {e}")  # -> write error if query fails
            raise

    def close(self):
        self.conn.close()
        logging.info("Databse closed")  # -> write in diary we closed databse
import json
from openai import OpenAI
class AIClient:
    def __init__(self, config_path: str = 'config.json'):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.api_key = config.get('openai_api_key')
                if not self.api_key:
                    raise ValueError("OpenAI key not found in config.json")  # -> stop if no api key
            self.client = OpenAI(api_key=self.api_key)  # -> set up openai for ai stuff
            logging.info("AIClient set up sucessfully")
        except Exception as e:
            logging.error(f"Failed to set up AIClient: {e}")  # -> write error if ai setup fails
            raise

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )  # -> ask openai to make text
            result = response.choices[0].message.content
            logging.info("Got answer from OpenAI")
            return result
        except Exception as e:
            logging.error(f"Failed to get answer from OpenAI: {e}")  # -> write error if ai fails
            raise
import os
import pandas as pd
def generate_user_profile(user_data: dict) -> dict:
    try:
        age = user_data.get('age', 30)
        interests = user_data.get('interests', [])
        budget = user_data.get('budget', 1000)
        location = user_data.get('location', 'USA')
        visas = user_data.get('visas', [])

        # decide what kind of travel user likes
        if 'history' in interests or 'culture' in interests:
            travel_type = 'historical'  # -> user likes historical places
        elif 'nature' in interests or 'beach' in interests:
            travel_type = 'nature'  # -> user likes nature places
        elif 'hiking' in interests or 'sports' in interests:
            travel_type = 'adventure'  # -> user likes adventure
        else:
            travel_type = 'historical'

        profile = {
            'age': age,
            'interests': interests,
            'budget': budget,
            'location': location,
            'visas': visas,
            'travel_type': travel_type
        }
        logging.info("Made user profile")  # -> write in diary we made profile
        return profile
    except Exception as e:
        logging.error(f"Failed to make profile: {e}")  # -> write error if profile fails
        raise
import requests
def fetch_recommendations_from_db(travel_type: str) -> list:
    try:
        db = DatabaseManager()
        query = """
        SELECT Id, Name, TravelType, Rating, Location, Season, AvgCost, PopularityScore
        FROM Destinations
        WHERE TravelType = ?
        ORDER BY Rating DESC, PopularityScore DESC
        """  # -> get places from database that match travel type
        rows = db.fetch_data(query, (travel_type,))
        recommendations = []
        for row in rows:
            recommendations.append({
                'id': row[0],
                'name': row[1],
                'travel_type': row[2],
                'rating': row[3],
                'location': row[4],
                'season': row[5],
                'avg_cost': row[6],
                'popularity_score': row[7]
            })
        db.close()
        logging.info(f"Got {len(recommendations)} recomendations")  # -> write how many places we found
        return recommendations
    except Exception as e:
        logging.error(f"Failed to get recomendations: {e}")  # -> write error if we cant get places
        raise

def optimize_trip_plan(destinations: list, budget: int, days: int) -> dict:
    try:
        ai = AIClient()
        affordable_destinations = [d for d in destinations if d['avg_cost'] <= budget]  # -> only keep places we can afford
        if not affordable_destinations:
            return {"plan": "Sorry, no places fit your budget."}

        prompt = f"""
        I have {days} days and a budget of {budget}. Here are some places:
        {', '.join([d['name'] + ' (' + d['season'] + ')' for d in affordable_destinations])}.
        Make a trip plan for me.
        """
        plan = ai.generate_text(prompt)  # -> ask ai to make a trip plan
        logging.info("Made trip plan with AI")
        return {"plan": plan}
    except Exception as e:
        logging.error(f"Failed to make trip plan: {e}")  # -> write error if plan fails
        raise

def analyze_reviews_with_ai(destination_id: int) -> dict:
    try:
        db = DatabaseManager()
        query = "SELECT ReviewText FROM Reviews WHERE DestinationId = ?"  # -> get reviews for this place
        rows = db.fetch_data(query, (destination_id,))
        reviews = [row[0] for row in rows]
        db.close()

        if not reviews:
            return {"sentiment": "No reviews found"}

        ai = AIClient()
        prompt = f"""
        Analyze these reviews: {', '.join(reviews)}.
        Tell me the overall sentiment (positive, negative, or neutral).
        """
        sentiment = ai.generate_text(prompt)  # -> ask ai to check revievs
        logging.info(f"Checked revievs for destination {destination_id}")
        return {"sentiment": sentiment}
    except Exception as e:
        logging.error(f"Failed to check revievs: {e}")  # -> write error if reveiw check fails
        raise

def get_travel_insights_across_seasons(location: str) -> dict:
    try:
        db = DatabaseManager()
        query = """
        SELECT r.Season, AVG(r.Rating) as AvgRating, COUNT(r.Id) as ReviewCount
        FROM Destinations d
        JOIN Reviews r ON d.Id = r.DestinationId
        WHERE d.Location = ?
        GROUP BY r.Season
        """  # -> get season info from database
        rows = db.fetch_data(query, (location,))
        db.close()

        insights = {}
        best_season = None
        best_score = 0
        for row in rows:
            season = row[0]
            avg_rating = row[1]
            review_count = row[2]
            insights[season] = f"Average rating: {avg_rating:.1f}, Reviews: {review_count}"
            if avg_rating > best_score:
                best_score = avg_rating
                best_season = season

        if best_season:
            insights['best_season'] = best_season  # -> save the best season
        logging.info(f"Got season insights for {location}")
        return insights
    except Exception as e:
        logging.error(f"Failed to get season insights: {e}")  # -> write error if season check fails
        raise

def generate_visual_report(data: dict):
    import matplotlib.pyplot as plt
    try:
        recommendations = data.get('recommendations', [])
        if len(recommendations) < 1:
            return

        top_destinations = sorted(recommendations, key=lambda x: x['rating'], reverse=True)[:5]  # -> take top 5 places by rating
        names = [d['name'] for d in top_destinations]
        ratings = [d['rating'] for d in top_destinations]

        plt.figure(figsize=(8, 6))
        plt.bar(names, ratings, color='skyblue')  # -> make a bar chart for top places
        plt.xlabel('Destinations')
        plt.ylabel('Rating')
        plt.title('Top 5 Recomended Destinations')
        plt.savefig('static/top_destinations.png')  # -> save the chart
        plt.close()
        logging.info("Made visual report chart")
    except Exception as e:
        logging.error(f"Failed to make visual report: {e}")  # -> write error if chart fails
        raise

from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            user_data = {
                'age': int(request.form.get('age', 30)),
                'interests': request.form.get('interests', '').split(','),
                'budget': int(request.form.get('budget', 1000)),
                'location': request.form.get('location', 'USA'),
                'visas': request.form.get('visas', '').split(',')
            }
            days = int(request.form.get('days', 7))

            profile = generate_user_profile(user_data)  # -> make user profile
            travel_type = profile['travel_type']
            recommendations = fetch_recommendations_from_db(travel_type)  # -> get places from database

            trip_plan = optimize_trip_plan(recommendations, profile['budget'], days)  # -> make trip plan

            if recommendations:
                review_analysis = analyze_reviews_with_ai(recommendations[0]['id'])  # -> check reviewss for first place
            else:
                review_analysis = {"sentiment": "No places to analyze"}

            seasonal_insights = get_travel_insights_across_seasons(profile['location'])  # -> check best season

            generate_visual_report({'recommendations': recommendations})  # -> make chart for report
            from jinja2 import Environment, FileSystemLoader
            env = Environment(loader=FileSystemLoader('templates'))
            template = env.get_template('report.html')
            with open('report.html', 'w') as f:
                f.write(template.render(
                    recommendations=recommendations,
                    trip_plan=trip_plan['plan'],
                    seasonal_insights=seasonal_insights
                ))  # -> save the html report
            logging.info("Made html report")
            return app.send_static_file('report.html')
        except Exception as e:
            logging.error(f"Web app error: {e}")
            return f"Error: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)  