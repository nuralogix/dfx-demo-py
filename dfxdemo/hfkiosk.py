import asyncio
import pygame
from quart import Quart, jsonify
from hfdemo import run_job
from pygametest import DemoAsync
import aiohttp

app = Quart(__name__)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((540, 960))
font = pygame.font.SysFont("Arial", 48)

# Concurrency lock
job_lock = asyncio.Lock()

# Queue for UI jobs
ui_queue = asyncio.Queue()

CALLBACK_URL = "http://example.com/callback"  # replace with your API endpoint


# -------------------------
# Fake SDK job
# -------------------------
async def run_job_test():
    await DemoAsync(screen,True)
    return "ok"

async def run_and_callback():
    async with job_lock:
        try:
            result = await run_job_test()
        except Exception as e:
            result = {"status": "error", "message": str(e)}

        # Perform callback
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(CALLBACK_URL, json=result)
            except Exception as e:
                print(f"Callback failed: {e}")

# -------------------------
# Quart endpoint
# -------------------------
@app.route("/start")
async def start_task():
    if job_lock.locked():
        return jsonify({"error": "Server is busy"}), 409

    # Schedule background job
    asyncio.create_task(run_and_callback())
    return jsonify({"accepted": True}), 202


# -------------------------
# UI loop (master loop)
# -------------------------
async def ui_loop():
    running = True
    idle_color = (30, 30, 30)
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        if not job_lock.locked():
            # idle screen
            screen.fill(idle_color)
            text = font.render("Idle...", True, (200, 200, 200))
            screen.blit(text, (50, 400))
            pygame.display.flip()

        clock.tick(30)
        await asyncio.sleep(0)  # yield back to loop

    pygame.quit()


async def main():
    server_task = asyncio.create_task(app.run_task("0.0.0.0", 5012))
    ui_task = asyncio.create_task(ui_loop())

    # Wait until either UI or server exits
    done, pending = await asyncio.wait(
        [server_task, ui_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel the other task(s) cleanly
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

if __name__ == "__main__":
    import contextlib
    asyncio.run(main())
