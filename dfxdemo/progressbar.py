import pygame
import math

class ImageProgressBar:
    def __init__(self, empty_image_path, full_image_path, padding_px = 0):
        """
        Progress bar that composites two images.

        :param empty_image_path: path to empty bar image
        :param full_image_path: path to full bar image
        """
        self.empty_img = pygame.image.load(empty_image_path).convert_alpha()
        self.full_img = pygame.image.load(full_image_path).convert_alpha()

        if self.empty_img.get_size() != self.full_img.get_size():
            raise ValueError("Empty and full bar images must have the same size")

        self.width, self.height = self.empty_img.get_size()
        self.percent = 0.0
        self.padding_px = padding_px
        self.inner_width = self.width-self.padding_px*2

    def set_percent(self, percent: float):
        """Set progress percent (0.0 to 1.0 or 0 to 100)."""
        self.percent = max(0.0, min(1.0, percent))

    def draw(self, surface, pos):
        """
        Draw the progress bar onto a target surface.

        :param surface: pygame.Surface to draw onto
        :param pos: (x, y) top-left position
        """
        # First blit the empty background
        surface.blit(self.empty_img, pos)

        # Compute visible width of the filled portion
        fill_w = int(self.inner_width * self.percent)
        
        # Clip the full image
        clip_rect = pygame.Rect(0, 0, self.padding_px + fill_w, self.height)
        surface.blit(self.full_img, pos, clip_rect)


# ---------- Demo -------------------------------------------------------------
def main():
    pygame.init()
    try:
        W, H = 1080, 300
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Clock Spinner – alternating opacity")
        clock = pygame.time.Clock()

        bar = ImageProgressBar("ui/bar_empty.png","ui/bar_full.png",170)

        running = True
        t = 0
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

            screen.fill((18, 18, 22))

            t = t + 0.01
            bar.draw(screen, (0, 0))
            percents = math.cos(t)/2.0 + 0.5

            bar.set_percent(percents)
            # Optional label
            # (simple text without AA for minimal deps)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
