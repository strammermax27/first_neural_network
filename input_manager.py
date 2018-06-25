import pygame
import thread
import time

pygame.init()
eta = 0.
key_pressed = False

FPS = 20
fpsClock = pygame.time.Clock()

displaysurf = pygame.display.set_mode((100,100), 0, 0)

def check_eta():
    global eta
    global key_pressed
    
    while 1:
        for event in pygame.event.get(): # event handling loop
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    eta *= 10.
                if event.key == pygame.K_DOWN:
                    eta /= 10.

        fpsClock.tick(FPS)
        pygame.display.update() 

thread.start_new_thread(check_eta, () )






