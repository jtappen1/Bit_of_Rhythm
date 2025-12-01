# Bit_of_Rhythm -- CIS 5810 Final Project 
### Created by Ben Kurzion and John Tappen

## Project Description
We sought to create a tool that, given a video of a drummer drumming, can transcribe the rhythms played and output sheet music. 

We fine-tuned a YOLOv12 model with a hand labeled dataset to track the drumsticks in the video, and captured stick-drum collisions using a Kalman Filter. These collisions represent notes. 

Finally, we took the sequence of collision timestamps and converted them into sheet music using some python libraries and some light HTML.
