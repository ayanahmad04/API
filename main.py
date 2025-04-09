from movie_recommendation import app
from flask_ngrok import run_with_ngrok

# Uncomment the next line if you want to use ngrok
# run_with_ngrok(app)

if __name__ == "__main__":
    # For local development with debug mode on port 8080
    app.run(debug=True, port=8080)
    
    # For production (without debug mode)
    # app.run()