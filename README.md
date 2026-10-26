# Say-and-Grasp

Tell a low-cost robot arm what to do in plain language, like *"put the red cup next to the blue block"*, and it does it. You don't record new demonstrations or train a new policy for each object.

Built for the **Meta Global AI Developer Hackathon** (Oct 26 – Nov 4, 2026), Indie Developer track, *Push the Frontier* theme.

## How it works

1. **Plan:** Muse Spark breaks the instruction into steps and names the target objects.
2. **See:** SAM 3.1 segments each named object in the camera feed from a short text prompt.
3. **Grasp:** each mask becomes a grasp target (centroid plus gripper yaw across the object's short side), which is mapped to table coordinates and sent to the arm.
4. **Check:** Muse Spark looks at the camera frame after each step and either confirms success or retries or re-plans.

Hardware: SO-101 arm (LeRobot), a wrist camera and an overhead USB camera, all run from a laptop with no GPU. The models run through Meta Model API.

## Status

Work in progress during the hackathon.

- [x] SAM 3.1 client: text prompt → masks → grasp point and yaw (`sam_client.py`)
- [ ] Overhead camera calibration (pixels → table coordinates)
- [ ] Arm control: move to the grasp target and pick
- [ ] Muse Spark planner and step verification loop
- [ ] Voice commands (Muse Voice Transcribe)
- [ ] Evaluation against an imitation-learning baseline

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export META_API_KEY=your-key   # from dev.meta.ai → API keys

# Segment an object and compute its grasp point (camera index or image file)
python sam_client.py --camera 0 --prompt "blue lego block"
python sam_client.py --image photo.jpg --prompt "red cup"
```

The script prints each detected object's box and grasp point, and saves an overlay image showing the masks, with a red line marking where the gripper jaws close.

## What existed before the hackathon

To keep the judging clear, here is what came before the event: the SO-101 arm hardware and its LeRobot setup (calibration, teleoperation, data recording), and an ACT imitation-learning policy trained on a block-picking dataset. The ACT policy is used here only as a comparison baseline.

Everything in this repository was written during the hackathon.

## License

MIT
