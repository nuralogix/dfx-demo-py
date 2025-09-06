import asyncio
import cv2
import math
import pygame
import numpy as np
from kettuspinner import KettuSpinner
from progressbar import ImageProgressBar
from clockspinner import ClockSpinner
from pygame.locals import *
from dfxutils.app import *
from lerp import lerp
from dfxutils.opencvhelpers import CameraReader
from pygame import Surface, gfxdraw
from kioskrequest import KioskRequest

defaultCam = 1

class DemoRenderer():
    def __init__(self, app, _screen : pygame.Surface):
        self._app = app
        self._screen = _screen
        self.W = _screen.get_width()
        self.H = _screen.get_height()
    
        scale = 0.5
        self.lerpSpd = 0.1
        
        self.frame_number = 0
        
        self.font = pygame.font.SysFont("Poppins", 100)
        self.surface = pygame.Surface([1080, 1920])
        self.background = pygame.image.load('ui/background.png')
        self.ui_error_background = pygame.image.load('ui/background.png')
        self.ui_waiting_results = pygame.image.load('ui/processing.png')
        #self.ui_frame = pygame.image.load('ui/frame.png').convert_alpha()
        self.ui_frame_top = pygame.image.load('ui/frame_top.png').convert_alpha()
        self.ui_frame_bottom = pygame.image.load('ui/frame_bottom.png').convert_alpha()
        self.ui_camera = pygame.image.load('ui/camera.png').convert_alpha()
        self.ui_intro = pygame.image.load('ui/intro.png').convert_alpha()
        self.ui_completed = pygame.image.load('ui/completed.png').convert_alpha()
        self.ui_inner = pygame.image.load('ui/inner_circle.png').convert_alpha()
        self.ui_user_started = pygame.image.load('ui/user_started.png').convert_alpha()
        self.ui_not_ready = pygame.image.load('ui/not_ready.png').convert_alpha()
        self.num_results_received = 0
        self.frame_extension = 400
        self.last_surface = pygame.Surface([1080, 1920])
        self.bar = ImageProgressBar("ui/bar_empty.png","ui/bar_full.png",170)

        self.alphas = {}

        for step in MeasurementStep:
            self.alphas[step] = 0.0

        self.spinner = ClockSpinner(
                    radius=770,
                    spokes=64,
                    line_len=330,
                    chunks=6,
                    line_thickness=30,
                    speed_hz=0.4,
                    wave_len=20,
                    color_incomplete=(255, 255, 255),
                    color_complete=(137, 206, 229),
                    color_results=(42, 127, 156),
                )
        # create a text surface object,
        # on which text is drawn on it.
        self.text = self.font.render('Testitesti', True, [255,255,255])

        # create a rectangular object for the
        # text surface object
        self.textRect = self.text.get_rect()

    
    def SetStep(self,step):
        self._app.step = step

    def SetFrameNumber(self,frame_number):
        if (frame_number != self.frame_number):
            self.frame_number = frame_number
            self.frame_lerp = 1

    def SetResults(self, number_results):
        self.num_results_received = number_results
        self.spinner.setResults(number_results)

    def SetChunksSent(self,number_chunks_sent):
        self._app.number_chunks_sent = number_chunks_sent

    def Start(self):
        self.cap = cv2.VideoCapture(defaultCam)

    def Release(self):
        self.cap.release()

    def RenderSurface(self,frame_surface:pygame.Surface):
        scaled_cam = pygame.transform.scale(frame_surface, (2400, 1350))
        self.surface.blit(scaled_cam, (0, 360), area = Rect(600,0,1500,1500))
        self.Render()

    def RenderWithCamera(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # cv2 is BGR, pygame expects RGB
        frame = np.rot90(frame)                          # optional: rotate if needed
        frame_surface = pygame.surfarray.make_surface(frame)
        #scaled_cam = pygame.transform.scale(frame_surface, (2400, 1350))
        #self.surface.blit(scaled_cam, (0, 360), area = Rect(600,0,1500,1500))
        self.surface.blit(frame_surface,(0,0))
        self.Render()

    async def FadeOut(self):
        clock = pygame.time.Clock()
        for r in range(255):
            self.last_surface.set_alpha(255-r)
            self._screen.fill([0,0,0])
            self._screen.blit(self.last_surface,(0,0))
            pygame.display.flip()
            await asyncio.sleep(0)
            clock.tick(60)

    def Render(self):
        
        t = pygame.time.get_ticks() / 1000.0
        framesTotal = (self._app.end_frame-self._app.begin_frame) +1
        if framesTotal > 0:
            percents = self.frame_number / framesTotal


        
        transitionInProgress = False
        for r in MeasurementStep:
            if r != self._app.step:
                
                self.alphas[r] = lerp(self.alphas[r], 0, self.lerpSpd)
                if self.alphas[r] > 0.3:
                    transitionInProgress = True
        
        if transitionInProgress == False:
            self.alphas[self._app.step] = lerp(self.alphas[self._app.step], 1, self.lerpSpd)
        
        
    
        self.spinner.current_chunk = self._app.number_chunks_sent

        self.ui_not_ready.set_alpha(self.alphas[MeasurementStep.NOT_READY]*255)
        self.ui_user_started.set_alpha(self.alphas[MeasurementStep.USER_STARTED]*255)
        self.ui_completed.set_alpha(self.alphas[MeasurementStep.COMPLETED]*255)
        self.ui_waiting_results.set_alpha(self.alphas[MeasurementStep.WAITING_RESULTS]*255)
        self.ui_error_background.set_alpha(self.alphas[MeasurementStep.FAILED]*255)
        self.ui_inner.set_alpha(self.alphas[MeasurementStep.MEASURING]*255)

        if (self._app.step == MeasurementStep.COMPLETED):
            self.surface.blit(self.background, (0, 0))

        if (self.alphas[MeasurementStep.NOT_READY] > 0):
            self.surface.blit(self.ui_not_ready, (0, 0))
        if (self.alphas[MeasurementStep.COMPLETED] > 0):
            
            self.surface.blit(self.ui_completed, (0, 0))
        if (self.alphas[MeasurementStep.USER_STARTED] > 0):
            self.surface.blit(self.ui_user_started, (0, 0))

        if (self.alphas[MeasurementStep.WAITING_RESULTS] > 0):
            self.surface.blit(self.ui_waiting_results, (0, 0))

        if (self.alphas[MeasurementStep.FAILED] > 0):
            self.surface.blit(self.ui_error_background, (0, 0))

        if (self.alphas[MeasurementStep.MEASURING]>0):
            self.surface.blit(self.ui_inner, (0, 0))

        if (self._app.step in [MeasurementStep.MEASURING, MeasurementStep.WAITING_RESULTS]):
            self.spinner.draw(self.surface, (540, 960), t,percents,self.alphas[MeasurementStep.MEASURING])
            
        
        targetExtension = 290
        if self._app.step == MeasurementStep.NOT_READY:
            targetExtension = 290
        elif self._app.step == MeasurementStep.USER_STARTED:
            targetExtension = 240
        elif self._app.step == MeasurementStep.MEASURING:
            targetExtension = 300
        elif self._app.step == MeasurementStep.COMPLETED:
            targetExtension = 750
        
        self.frame_extension = lerp(self.frame_extension,targetExtension,0.1)
        
        self.surface.blit(self.ui_frame_top, (0, -self.frame_extension))
        self.surface.blit(self.ui_frame_bottom, (0, 1132 + self.frame_extension))
        #if (self._app.step == MeasurementStep.WAITING_RESULTS):
        #        s§§elf.bar.draw(self.surface, (0, 1678))

        
        
        #self.bar.set_percent(percents)
        scaled = pygame.transform.scale(self.surface, (self.W, self.H))
        self.last_surface = scaled
    
        self._screen.blit(scaled, scaled.get_rect())
        pygame.display.flip()
 


async def DemoAsync(screen: Surface, autonomous = False, data: KioskRequest = None):
    kettu = KettuSpinner("/Users/alexkivikoski/Documents/hfkiosk/kettuanim")
    await kettu.FadeInOut(screen,(screen.get_width()/2-kettu.width/2,screen.get_height()/2-kettu.height/2))
    results = {}
    running = True
    
    

    app = AppState()
    app.end_frame = 900
    app.number_chunks = 6
    
    renderer = DemoRenderer(app,screen)

    renderer.Start()
    
    t = 0


    while running:
        # --- get frame from cv2 ---
        if autonomous:
            step = MeasurementStep.NOT_READY
            if (t > 24): running = False
            if (t > 20): step = MeasurementStep.COMPLETED
            elif (t > 10): step = MeasurementStep.MEASURING
            elif (t > 4): step = MeasurementStep.USER_STARTED
            renderer.SetStep(step)

        t = t + 0.03
        
        percents = (t % 10)/10
        
        


        # --- event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_1:
                    renderer.SetStep(-1)
                elif event.key == pygame.K_2:
                    renderer.SetStep(MeasurementStep.NOT_READY)
                elif event.key == pygame.K_3:
                    renderer.SetStep(MeasurementStep.USER_STARTED)
                elif event.key == pygame.K_4:
                    t = 0
                    renderer.SetStep(MeasurementStep.MEASURING)
                elif event.key == pygame.K_5:
                    renderer.SetStep(MeasurementStep.WAITING_RESULTS)
                elif event.key == pygame.K_6:
                    renderer.SetStep(MeasurementStep.COMPLETED)
                elif event.key == pygame.K_7:
                    renderer.SetStep(MeasurementStep.USER_CANCELLED)
                elif event.key == pygame.K_8:
                    renderer.SetStep(MeasurementStep.FAILED)


        renderer.SetChunksSent(math.floor(percents*renderer.spinner.chunks))
        renderer.SetResults(max(0,math.floor(percents*renderer.spinner.chunks)-1))
        renderer.SetFrameNumber(percents*app.end_frame);
        renderer.RenderWithCamera()
        await asyncio.sleep(0)

    results["t"] = t
    renderer.Release()
    await renderer.FadeOut()
    return results


if __name__ == "__main__":
    pygame.init()
    W, H = 1080*0.5, 1920*0.5
    screen = pygame.display.set_mode((W,H))
    #imreader = CameraReader(1, mirror=False, fps=30, width=W,height=H)
    asyncio.run(DemoAsync(screen,True))
    pygame.quit()
