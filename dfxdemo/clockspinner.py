import math
import pygame

# ---------- Spinner ----------------------------------------------------------
class ClockSpinner:
    def __init__(
        self,
        radius=60,
        spokes=12,
        wave_len = 1.0,
        line_len=40,
        line_thickness=6,
        chunks = 6, 
        speed_hz=1.2,          # how fast the opacity alternates
        color_incomplete=(255, 255, 255), # RGB (alpha animated)
        color_complete=(50, 255, 120),
        color_results=(247, 237, 127),
        bg_alpha=0             # optional background alpha for the spinner surface
    ):
        self.radius = radius
        self.spokes = spokes
        self.line_len = line_len
        self.line_thickness = line_thickness
        self.speed_hz = speed_hz
        self.color_incomplete = color_incomplete
        self.color_complete = color_complete
        self.current_chunk = 0
        self.wave_len = wave_len
        self.chunks = chunks
        self.color_results = color_results
        self.spokesPerChunk = spokes / float(chunks)
        self.frame_lerp = 0
        self.completed_chunks = 0

        size = (radius + line_len + line_thickness) * 2
        self.size = size
        self.prev_spokesComplete = 0
        self.surface = pygame.Surface((size, size), pygame.SRCALPHA)
        if bg_alpha:
            self.surface.fill((0, 0, 0, bg_alpha))

    def setResults(self, results):
        self.completed_chunks = results

    def draw(self, target_surface, center_pos, t_seconds: float, progress: float, transitionProgress: float = 1):
        """
        Draw the spinner at time t_seconds (float).
        The opacity of adjacent spokes alternates in opposite phase.
        """
        # Clear spinner surface each frame
        self.surface.fill((0, 0, 0, 0))

        cx = cy = self.size // 2
        angle_step = (2 * math.pi) / self.spokes 

        # Opacity animation: 0..255
        # Adjacent spokes are π out of phase -> alternating opacity
        base = 0        # minimum alpha
        amp = 159        # amplitude (base + amp ≤ 255)
        omega = 2 * math.pi * self.speed_hz

        spokesComplete = round(self.spokes*(progress))
        if (spokesComplete != self.prev_spokesComplete):
            self.prev_spokesComplete = spokesComplete
            self.frame_lerp = 1

        self.frame_lerp = max(0,self.frame_lerp - 0.1)

        for i in range(self.spokes):
            # phase alternates between 0 and π for neighbors
            stepCompleted = i < spokesComplete
            thisChunk = math.floor(i/self.chunks)
            chunkActive = i >= self.current_chunk*self.spokesPerChunk and i < (self.current_chunk+1)*self.spokesPerChunk
            

            phase = -i * math.pi / self.wave_len
            if (self.completed_chunks > thisChunk):
                a = base + amp * (0.5 + 0.5 * math.sin(omega/2.0 * t_seconds + phase))
                minalpha = 0
            elif (chunkActive):
                a = base + amp * (0.5 + 0.5 * math.sin(4 * omega * t_seconds + phase))
                minalpha = 150 if stepCompleted else 100
            else:
                a = base + 0.5 * amp * (0.5 + 0.5 * math.sin(omega * t_seconds + phase))
                minalpha = 150 if stepCompleted else 0

            if i == spokesComplete:
                minalpha = 200
            
            alpha = max(minalpha, min(255, int(a)))

            # Angle for this spoke (visual placement around circle)
            theta = i * angle_step - math.pi / 2  # start at 12 o'clock

            spokePercentage = i/self.spokes

            transitionInSpokes = (transitionProgress*1.1)*self.spokes


            alpha_reduction = 0 #(255-alpha)/3.0

            reduction = self.line_len if i > transitionInSpokes else 0

            x0 = cx + math.cos(theta) * (self.radius - reduction)
            y0 = cy + math.sin(theta) * (self.radius - reduction)
            x1 = cx + math.cos(theta) * (self.radius - self.line_len + alpha_reduction)
            y1 = cy + math.sin(theta) * (self.radius - self.line_len + alpha_reduction)

            thickness = int(self.line_thickness*alpha/255)

            if (self.completed_chunks > thisChunk):
                color = (*self.color_results,alpha)
                thickness = thickness + 3
            elif (stepCompleted):
                color = (*self.color_complete,alpha)
            else:
                color = (*self.color_incomplete, alpha)

            # Draw onto the alpha surface so per-line alpha works everywhere
            pygame.draw.line(
                self.surface,
                color,
                (x0, y0),
                (x1, y1),
                thickness
            )

        pygame.draw.circle(self.surface,(*self.color_incomplete, self.frame_lerp*55),(cx, cy),self.radius-self.line_len,8)
        # Blit centered at center_pos
        rect = self.surface.get_rect(center=center_pos)
        target_surface.blit(self.surface, rect)


# ---------- Demo -------------------------------------------------------------
def main():
    pygame.init()
    try:
        W, H = 600, 600
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Clock Spinner – alternating opacity")
        clock = pygame.time.Clock()

        spinner = ClockSpinner(
            radius=570,
            spokes=64,
            line_len=348,
            line_thickness=20,
            speed_hz=0.4,
            wave_len=20,
            chunks=6, 
            color_incomplete=(255, 255, 255),
            color_complete=(137, 206, 229),
            color_results=(42, 127, 156)
        )
        spokePercentage = 0
        running = True
        progress = 0
        results = 0
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

            screen.fill((18, 18, 22))
            spokePercentage = spokePercentage + (1-spokePercentage)*0.01
            t = pygame.time.get_ticks() / 1000.0

            progress = (t % 10)/10
            results = math.floor(progress*progress*spinner.chunks)
            spinner.setResults(results)
            spinner.current_chunk = math.floor(progress*spinner.chunks)
            

            spinner.draw(screen, (W // 2, H // 2), t, progress,spokePercentage)

            # Optional label
            # (simple text without AA for minimal deps)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
