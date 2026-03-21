# Project instructions

Build a local-only Python application for autonomous drone simulation in AirSim.

## Goal
Create an MVP that connects to AirSim, controls a multirotor drone, reads RGB and depth images, detects an ArUco marker, centers the marker in the frame, approaches it, avoids simple frontal obstacles using depth data, and performs precision landing on the marker.

## Hard constraints
- Local-only application
- No database
- No web frontend
- No Docker for MVP
- No ROS 2 for MVP
- No YOLO or heavy ML for MVP
- Use AirSim, OpenCV, asyncio, NumPy, PyYAML
- Keep code modular
- Keep all magic numbers in config/settings.yaml
- Every step must leave the project runnable
- Do not put all logic into one file
- Do not replace working code with pseudo-code
- Do not do speculative refactors unless requested

## Required project structure
drone_cv/
  README.md
  requirements.txt
  config/
    settings.yaml
  app/
    main.py
    bootstrap.py
  adapters/
    airsim_client.py
  vision/
    frame_fetcher.py
    aruco_detector.py
    depth_analyzer.py
    overlays.py
  control/
    pid.py
    visual_servo.py
    obstacle_avoidance.py
    landing_controller.py
  mission/
    states.py
    mission_manager.py
    search_pattern.py
  telemetry/
    models.py
    logger.py
    recorder.py
  tests/
    test_pid.py
    test_aruco_detector.py
    test_depth_analyzer.py

## Architecture rules
- Keep AirSim API calls only in adapters/airsim_client.py
- Keep CV logic only in vision/*
- Keep control logic only in control/*
- Keep mission state machine only in mission/*
- Use asyncio tasks for camera loop, telemetry loop, vision loop, mission loop, and control loop
- Use typed dataclasses or pydantic models
- Add logging
- Add acceptance checks after each milestone

## Delivery rules for every task
For each task:
1. Explain what you will change
2. Create or modify files
3. Show exact run commands
4. Show a short acceptance checklist
5. Keep the project runnable