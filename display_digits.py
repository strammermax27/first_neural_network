import pygame



class Display():
    
    def __init__(self):
        global FPSCLOCK
        global DISPLAYSURF
        global BOXSIZE

        pygame.init()
        FPSCLOCK = pygame.time.Clock()
        DISPLAYSURF = pygame.display.set_mode((280, 280))
        pygame.display.set_caption('Handwritten Digit')
        BOXSIZE = 10
        DISPLAYSURF.fill((1,1,255))


    def draw_image(self, handwritten_digit):
        global DISPLAYSURF
        global FPSCLOCK
        global BOXSIZE

        DISPLAYSURF.fill((0,0,0))
        
        i = 0
        for pixel in handwritten_digit:
            i += 1
            colum = i%28
            line = i/28 
            
            left = colum * BOXSIZE
            top  = line  * BOXSIZE
            
            color = (pixel*255, pixel*255, pixel*255)
            
            pygame.draw.rect(DISPLAYSURF, color, (left, top, BOXSIZE, BOXSIZE))
            
            

        pygame.display.update()
        #pygame.time.wait(3000)
    
    def image_correct(self, correct):

        global DISPLAYSURF
        global FPSCLOCK
        global BOXSIZE

        if correct: 
            DISPLAYSURF.fill((255,0,0))
        else:
            DISPLAYSURF.fill((255,0,0))

        pygame.display.update()

            
            

    def display_data(self, data):
        pass
        
