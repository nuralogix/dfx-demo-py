import asyncio
import aiohttp
import pygame
from quart import Quart, jsonify, request
from hfkiosk import DemoAsync
from kioskrequest import KioskRequest
from quart_schema import QuartSchema, validate_request
from attrs import define
from quart_schema import DataSource, validate_request
from hfdemo import run_job

app = Quart(__name__)

QuartSchema(app)
# -------------------------
# Pygame setup
# -------------------------
pygame.init()
screen = pygame.display.set_mode((540, 960))
pygame.display.set_caption("Kiosk App")
font = pygame.font.SysFont("Arial", 48)

# -------------------------
# Shared state
# -------------------------
job_lock = asyncio.Lock()
CALLBACK_URL = "http://example.com/callback"  # replace with your API endpoint

@app.before_serving
async def startup():
    #todo: call endpoint
    aa = 0

@app.after_serving
async def shutdown():
    #todo: call endpoint
    aa = 0


# -------------------------
# Job runner with callback
# -------------------------
async def run_and_callback(data: KioskRequest = None, real = False):
    async with job_lock:
        try:
            if (data is not None):
                data.log_user_demographics()

            if real:
                vitals = await run_job(screen,data)
            else:
                vitals = await DemoAsync(screen, True)
            # callback to server

            if data is not None:
                result = {
                    "deviceId": data.deviceId,
                    "requestId": data.requestId,
                    "content": vitals
                }
        except Exception as e:
            result = {
                "deviceId": data.deviceId,
                "requestId": data.requestId,
                "status": "error", "message": str(e)
            }

        if (result is not None):
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

@app.route("/joo", methods=["POST"] )
@validate_request(KioskRequest)
async def NotReally(data: KioskRequest):
    if job_lock.locked():
        return jsonify({"error": "Server is busy"}), 409
    
    
    asyncio.create_task(run_and_callback(data))
    return jsonify({"accepted": True}), 202

@app.route("/real", methods=["POST"] )
@validate_request(KioskRequest)
async def Really(data: KioskRequest):
    if job_lock.locked():
        return jsonify({"error": "Server is busy"}), 409
    
    
    asyncio.create_task(run_and_callback(data,True))
    return jsonify({"accepted": True}), 202
        

# -------------------------
# UI loop
# -------------------------
async def ui_loop():
    running = True
    idle_color = (0, 0, 0)
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
            #text = font.render("Idle...", True, (200, 200, 200))
            #screen.blit(text, (50, 400))
            pygame.display.flip()

        clock.tick(30)
        await asyncio.sleep(0)  # yield back to loop

    pygame.quit()

# -------------------------
# Main
# -------------------------
async def main():
    server_task = asyncio.create_task(app.run_task("0.0.0.0", 5012))
    ui_task = asyncio.create_task(ui_loop())
    done, pending = await asyncio.wait(
        [server_task, ui_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
