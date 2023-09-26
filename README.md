# STRIKE A POSE! 
**Predictive Body Pose Classification via Sequential Neural Network Analysis in a Dynamic Videogame**

## Intro 
"STRIKE A POSE!" is a videogame I developed in 1.5 weeks as my final project at [SPICED's](https://www.spiced-academy.com/en) Data Science Bootcamp.   

It's a fun workout challenge using body pose classification, in which to follow along with various exercises:   
From squats to standing, hiding, making an X, or striking a unique pose of your own, you'll work up a sweat! 🏋️   

You have the flexibility to adjust the number of rounds and the time between poses for an extra challenge.    

Data collection, model training, and the game script are modularized, making it easy to customize to meet your unique workout needs 💪  

### Demo
Turn the audio on! 🎧  
Check out the demonstration video in better quality [here](https://www.loom.com/share/c715db6d054c44cab8a703be838f9201?sid=b5d8c26a-da8b-4349-9043-285fca207493).

https://github.com/alx-sch/STRIKE_A_POSE_Body_Pose_Classification_Game/assets/134595144/bcffa499-bdbe-4177-996d-abcd805d37d4


### Installation

1. Clone this Git repository.
2. Optional: Create a new virtual environment to run the game.
```rb
python3 -m venv envname # to create a virtual environment (macOS/LINUX)
source envname/bin/activate # to activate it 
deactivate # to deactivate it
```
3. Install game-specific requirements.
```rb
pip install -r requirements.txt
```
### How to Play
To start the game, please specify the number of rounds and the time between poses:
```rb
python play.py <ROUNDS> <TIME>
python play.py 10 5 # 10 rounds with 5 seconds between poses
```

A window with a videofeed will open. Make sure that the background is rather neutral.
- Find the right position and camera angle before starting the countdown. Ensure that your feet and arms fill out the green square.
- Press 'Space' to start the countdown.
- Press 'R' to restart the script execution.
- Press 'Q' to quit the script.

### Make it Yours!
Adjust the game to your needs! Please refer to the documentation in collect_data.py, train_model.ipynb, and play.py for details.
- Collect training data of poses of your choice with **collect_data.py**.
- Train a new model to detect these new poses and test the model's performance with **train_model.ipynb** (feel free to play around and try out a different model's architecture! :) ).
- Adjust the model used, poses, and respective sound files (see [Audio Credits](#audio-credits)) in ** play.py** to start your new Pose Game! 

### Audio Credits
- Background music is sampled by [KevWest: Funky Disco Beat](https://www.looperman.com/loops/detail/332124/funky-disco-beat-free-123bpm-disco-drum-loop).
- Sound effects are downloaded from [Pixabay](https://pixabay.com/) (CC0 License).
- Commands and in-game speech are generated using the Python library gTTS (Google Text-to-Speech).

### Collaborate
Pull requests are more than welcome! 

### License
