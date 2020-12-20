# Purpose of this project
<p>
    Many old people who are granted a pension are required to produce a certificate in order to prove they're alive and then they recieve their pension.This project is to ease out the effort they put in by making them sit in front of a liveness detector model for a minute and then that model predicts the liveness of the person.
</p>

<p>
    This is a flask app which takes data like name and aadhar number and an image on registration and trains the model.On login, the user simply has to put name and aadhar and then continue to liveness detector which uses the webcam to feed in the data and predict. The liveness detector applies colored histogram concatenation and eye blinnk models to enure only real humans can authenticate into the system.
</p>

<p><b>app.py  is the flask app. Please use that to set your environment variable to run it and the flask environemnt should be development(Windows).</b></p>
