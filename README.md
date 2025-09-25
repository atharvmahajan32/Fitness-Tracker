# AI-Fitness-Trainer
This is a web application with following features:
1. Exercise form tracking.
2. Rep counting.
3. Tutorials of each exercise(both video as well as text based)
4. Daily Nutrition Tracker


## Home Page
- Contains a brief of what we offer.
- Contains a music playlist to pump up your workout regime.
- Also contains a podcast playlist(for guidance and motivation)


## Training Arena
Here's where the fun begins. You get to choose the exercise which you want to perform and our model tracks your form and reps.

Available Exercises:
- Left Bicep Curl
- Right Bicep Curl
- Squats
- Pushups
- Shoulder Press


## Nutrition Tracker
- The user can track his/her nutrition by listing out the food items they ate for a particular meal.
- This helps one to monitor their food diet and adjust their next meal accordingly.


Finally, a chatbot is also integrated which answers questions on fitness and fitness related queries.

## Tech Stack
- OpenCV: To allow access of web camera and convert the live feed to frames so that necessary operations can be performed.
- Mediapipe: To extract key-points from the human body which helps to calculate angles between different body parts.
- Stramlit: Python based framework to develop website.

## Note:
- Make sure to put in your email id to allow form submission.(Change to be made in 1_🏠_Home.py file)

## How to run
- Clone the repository
```
git clone https://github.com/atharvmahajan32/Fitness-Tracker.git
```
- Create a venv and install dependencies
```
pip install uv
uv init 
uv venv --python 3.9  (works stably on python 3.9) 
uv add -r requirements.txt
```
- Run the website
```
streamlit run 1_🏠_Home.py
```
Feel free to raise any issues/bugs/suggestions in the issues page of this repository.

Thank You!!
"# AI-Fitness-Tracker" 
