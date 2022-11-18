from gtts import gTTS
from pygame import mixer
import time


complete=gTTS(text="Learning completed.", lang="en")

complete.save("learning_complete.mp3")

mixer.init()
mixer.music.load("learning_complete.mp3")
mixer.music.set_volume(1.0)
mixer.music.play()
time.sleep(2.0)
