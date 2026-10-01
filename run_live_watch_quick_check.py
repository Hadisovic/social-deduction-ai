import threading
import time
import pygame
import live_watch_training

def send_esc():
    time.sleep(2.0)
    if pygame.get_init() and pygame.display.get_init():
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}))

t = threading.Thread(target=send_esc, daemon=True)
t.start()
live_watch_training.main()
