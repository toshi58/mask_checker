import os

def beep(freq, dur=100):
  os.system('play -n synth %s sin %s' % (dur/1000, freq))