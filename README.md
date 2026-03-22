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
  host: "auto"
  port: 41451
```
3. Place an ArUco marker in view of the configured camera if you want to test tracking and landing flows.

### WSL + Windows AirSim (no Unreal Engine required)

1. On Windows, download and run a prebuilt AirSim environment (for example `Blocks.exe`).
2. Keep `config/settings.yaml` with `airsim.host: "auto"`.
   In WSL this auto-resolves the Windows host IP from `/etc/resolv.conf`.
3. Run smoke test from WSL:
```bash
source .venv/bin/activate
python -m app.main --mode smoke
```
4. If smoke is successful, run full dev loop:
```bash
python -m app.main --mode dev
```

## Run The App

Always run commands after activating `.venv`:
```bash
source .venv/bin/activate
```

Local development mode:
```bash
python -m app.main --mode dev
```

Profiles:
```bash
python -m app.main --mode local --profile default
python -m app.main --mode local --profile safe
python -m app.main --mode local --profile ci
```

Profile behavior:
- `default`: use values from `config/settings.yaml`
- `safe`: tighten watchdog and freshness guards for more conservative runtime behavior
- `ci`: disable OpenCV windows and debug frame saving for automation-friendly execution

`dev` mode opens an OpenCV window by default and supports manual override:
- `A` / `D`: yaw left / right
- `W` / `S`: altitude up / down
- `I` / `K`: forward / backward
- `J` / `L`: strafe left / right
- `M`: toggle manual-only mode
- `Space`: pause/resume automatic yaw behavior
- `Q` or `Esc`: stop runtime and close the window

The OpenCV debug window for `dev` mode is currently disabled by default:
```yaml
dev_ui:
  enabled: false
```
Set `dev_ui.enabled: true` in `config/settings.yaml` if you want the camera debug window back.
When `dev_ui.enabled: false`, manual controls are read from the terminal where you launched the app.
In WSL terminals, use the control key followed by `Enter` for reliable input delivery.

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

Use `A` / `D` to rotate, `W` / `S` to move the drone higher or lower, `I` / `K` to move forward or backward, `J` / `L` to shift left or right, `M` to toggle manual-only mode, and `Space` to toggle the automatic spin in local mode.
Press `Q` or `Esc` to close it.
The same controls are also available from the terminal in `dev` mode when the OpenCV window is disabled.
In WSL terminals, enter the command key and press `Enter`.

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

Stable production-oriented regression suite:
```bash
python -m pytest tests/test_pid.py tests/test_mission_manager.py tests/test_precision_landing.py tests/test_profiles.py tests/test_settings_validation.py tests/test_command_safety.py tests/test_watchdog.py tests/test_freshness.py tests/test_logging.py tests/test_aruco_detector.py tests/test_depth_analyzer.py
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

The runtime now also includes:
- strict config validation at startup
- command safety clamping before control is applied
- loop heartbeat watchdog protection
- stale sensor-data protection
- optional structured JSON logs with `app.log_format: json`

By default, the main runtime does not auto-land when the marker is centered at low altitude:
```yaml
mission:
  auto_descend_on_target: false
  auto_land_on_target: false
```
This keeps the drone holding position near the target instead of descending or triggering landing automatically.

## Current Limitations

- AirSim calls are synchronous and are wrapped with `asyncio.to_thread` rather than using a native async AirSim client.
- The mission logic in the asyncio runtime is intentionally simple and does not yet replace every standalone demo script.
- Obstacle avoidance is basic left/center/right zone analysis from a single depth frame.
- Marker distance is estimated from image area only, not from full pose estimation.
- Precision landing is cautious but still MVP-grade and not hardened for noisy real-world sensing.
- Debug frame saving is implemented, but video recording is not included in the MVP.
- The project assumes a local simulator only. No ROS 2, no web frontend, no database, and no Docker integration are included.
