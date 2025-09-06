import asyncio
import pygame
import math

class KettuSpinner:
    def __init__(self,folderpath,size=256):
        """
        Progress bar that composites two images.

        :param empty_image_path: path to empty bar image
        :param full_image_path: path to full bar image
        """
        self.imgs = []
        self.frames = 29
        for n in range(self.frames):
            filename = f"{(n+1):04d}" + ".png"
            self.imgs.append(pygame.image.load(folderpath + "/" + filename))
        
        self.width = size
        self.height = size

    def draw(self, surface, pos,t):
        """
        Draw the progress bar onto a target surface.

        :param surface: pygame.Surface to draw onto
        :param pos: (x, y) top-left position
        """
        # First blit the empty background
        frame = int(t % self.frames)
        surface.blit(self.imgs[frame], pos)

    async def FadeInOut(self,surface,pos):
        clock = pygame.time.Clock()
        for r in range(255):
            frame = int(r % self.frames)
            img = self.imgs[frame]
            alpha = r if r < 128 else 128-(r-128)
            img.set_alpha(alpha*2)
            surface.fill([0,0,0])
            surface.blit(img,pos)
            pygame.display.flip()
            await asyncio.sleep(0)
            clock.tick(60)


# ---------- Demo -------------------------------------------------------------
def main():
    pygame.init()
    try:
        W, H = 300, 300
        screen = pygame.display.set_mode((W, H))
        clock = pygame.time.Clock()

        bar = KettuSpinner("/Users/alexkivikoski/Documents/hfkiosk/kettuanim")

        running = True
        t = 0
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

            screen.fill((18, 18, 22))

            t = t + 1
            bar.draw(screen, (0, 0),t)
           
            
            # Optional label
            # (simple text without AA for minimal deps)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
