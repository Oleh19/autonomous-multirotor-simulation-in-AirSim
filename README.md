# drone_cv
Local-only Python MVP for autonomous multirotor simulation in AirSim. The project connects to AirSim, reads RGB and depth images, detects an ArUco marker, centers on it, performs simple obstacle avoidance from depth, and supports cautious marker-guided descent and landing building blocks.

## Requirements

- Python 3.9 or newer
- AirSim running locally with a multirotor vehicle
- OpenCV-compatible system environment for `opencv-contrib-python`

## Setup

1. Create a virtual environment.
```bash
python3 -m venv .venv
```

2. Activate it.
```bash
source .venv/bin/activate
```

3. Upgrade `pip` inside `.venv`.
```bash
python -m pip install --upgrade pip
```

4. Install project dependencies.
```bash
python -m pip install -r requirements.txt
```

5. Review local settings if needed.
```bash
sed -n '1,260p' config/settings.yaml
```

## Quick Start

Full local setup and launch without AirSim:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m app.main --mode local
```

## Start AirSim

1. Launch an AirSim environment that contains a multirotor vehicle.
2. Confirm the AirSim RPC server is reachable at the host and port from `config/settings.yaml`.
   Default:
```yaml
airsim:
  host: "127.0.0.1"
  port: 41451
```
3. Place an ArUco marker in view of the configured camera if you want to test tracking and landing flows.

## Run The App

Always run commands after activating `.venv`:
```bash
source .venv/bin/activate
```

Local development mode:
```bash
python -m app.main --mode dev
```

Smoke mode:
```bash
python -m app.main --mode smoke
```

Local mode without AirSim:
```bash
python -m app.main --mode local
```

Local mode opens an OpenCV window by default. The window shows:
- left: synthetic camera view used by the vision pipeline
- right: a 2D top-down world view with the drone, marker, obstacle, heading, and current command

Use `A` / `D` to rotate, `W` / `S` to move the drone higher or lower, `J` / `L` to shift left or right, and `Space` to toggle the automatic spin in local mode.
Press `Q` or `Esc` to close it.

Frame capture debug:
```bash
python vision/frame_fetcher.py
```

Overlay debug:
```bash
python vision/overlays.py
```

Visual servo step:
```bash
python control/visual_servo.py
```

Obstacle avoidance step:
```bash
python control/obstacle_avoidance.py
```

Precision landing step:
```bash
python control/landing_controller.py
```

Mission state machine demo:
```bash
python mission/mission_manager.py
```

AirSim adapter smoke test:
```bash
python app/airsim_smoke_test.py
```

## Run Tests

All current tests:
```bash
python -m pytest tests
```

Individual test files:
```bash
python -m pytest tests/test_pid.py
python -m pytest tests/test_aruco_detector.py
python -m pytest tests/test_depth_analyzer.py
```

## Saved Outputs

Runtime artifacts:
- `artifacts/telemetry.jsonl`
- `artifacts/mission_events.jsonl`
- `artifacts/debug_frames/`

Other debug outputs:
- `debug_frames/`
- `debug_overlays/`

The runtime recorder saves key telemetry snapshots and mission events to JSONL files. Optional debug frames are enabled through:
```yaml
recording:
  output_dir: "artifacts"
  save_debug_frames: true
  debug_frame_interval_s: 1.0
```

## Runtime Overview

`python -m app.main --mode dev` runs separate asyncio tasks for:
- telemetry loop
- frame loop
- vision loop
- mission loop
- control loop

Tasks communicate through a shared typed runtime state protected by an `asyncio.Lock`. This keeps sensor ingestion, perception, decision-making, and actuation separated and easier to debug.

## Current Limitations

- AirSim calls are synchronous and are wrapped with `asyncio.to_thread` rather than using a native async AirSim client.
- The mission logic in the asyncio runtime is intentionally simple and does not yet replace every standalone demo script.
- Obstacle avoidance is basic left/center/right zone analysis from a single depth frame.
- Marker distance is estimated from image area only, not from full pose estimation.
- Precision landing is cautious but still MVP-grade and not hardened for noisy real-world sensing.
- Debug frame saving is implemented, but video recording is not included in the MVP.
- The project assumes a local simulator only. No ROS 2, no web frontend, no database, and no Docker integration are included.
