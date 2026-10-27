# Say-and-Grasp

**A third hand for your workbench.** You're mid-solder with both hands busy and say *"pass me the side cutters"*. A $250 robot arm finds them, picks them up and puts them down by your hand. Ask *"where's my multimeter?"* and it knows. Say *"tidy up"* and every tool goes back to its spot.

Built for the **Meta Global AI Developer Hackathon** (Oct 26 – Nov 4, 2026), Indie Developer track.

## Why

Anyone who builds hardware knows the moment: you're holding the iron and a wire, and the tool you need is just out of reach, or buried somewhere on the bench. Makers, repair technicians and electronics labs lose small chunks of time to this all day. For people who can't easily reach across a bench, it's a real barrier. Say-and-Grasp gives a low-cost arm the understanding to help, with no training for each tool.

## How it works

| Step | Meta model | What it does |
|---|---|---|
| Hear | Muse Voice Transcribe | Turns your spoken request into text |
| Understand | Muse Spark | Resolves what you mean ("the thing for cutting wire" → side cutters) and plans the steps |
| See | SAM 3.1 | Finds the tool from its name in the overhead camera, and tracks where every tool ends up |
| Act | — | Mask → grasp point and gripper angle → table coordinates (calibrated) → SO-101 inverse kinematics |
| Check | Muse Spark / Muse Spark Open | Looks at the result and confirms the handover worked, or retries |

Hardware: SO-101 arm (LeRobot) and an overhead USB camera, run from a laptop with no GPU. The models run through Meta Model API.

## Status

Work in progress during the hackathon.

- [x] SAM 3.1 client: text prompt → masks → grasp point and yaw
- [x] Overhead camera ↔ arm calibration
- [ ] Pick a named tool and place it at a spot (code written, testing on hardware)
- [ ] Voice requests and Muse Spark planning
- [ ] Tool memory: "where's my …?"
- [ ] Handover near your hand, and "tidy up"
- [ ] Success checks with retry, and timed trials

## Repo layout

```
src/say_and_grasp/
  sam.py           SAM 3.1 via Meta Model API: prompt → masks → grasp targets
  camera.py        single-frame capture from USB cameras
  calibration.py   pixel ↔ table homography, reachable zone
  kinematics.py    top-down IK on the SO-101 URDF (placo)
  arm.py           smooth, rate-limited arm and gripper control (LeRobot)
  cli/             command-line tools: segment, calibrate, pick
config/            per-setup calibration and home pose
assets/SO101/      SO-101 URDF + meshes (fetched, not committed)
outputs/           debug images (not committed)
```

## Quickstart

```bash
conda activate lerobot          # LeRobot env: arm control
pip install -e .                # installs the sag-* commands
export META_API_KEY=your-key    # dev.meta.ai → API keys

# SO-101 model for kinematics (once)
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git /tmp/soarm
cp -r /tmp/soarm/Simulation/SO101 assets/

sag-segment --camera 5 --prompt "side cutters"        # find a tool
sag-calibrate --camera 5 --marker "coin"              # camera ↔ arm calibration
sag-pick --save-home                                  # record a folded home pose
sag-pick --show-reach                                 # where the arm can reach
sag-pick --prompt "side cutters" --dry-run            # plan only
sag-pick --prompt "side cutters"                      # do it
```

## What existed before the hackathon

To keep the judging clear, here is what came before the event: the SO-101 arm hardware and its LeRobot setup (calibration, teleoperation). Everything in this repository was written during the hackathon.

## License

MIT
