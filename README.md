# Say-and-Grasp

**A voice-controlled third hand for your workbench.** Say *"Wiggle, pass me the side cutters"* and a $250 robot arm finds them, picks them up and drops them into your open hand, using Meta's Muse Spark and SAM 3.1 with **zero task-specific training**. Every handover is also recorded as a training episode and labelled by Muse Spark, so the robot builds its own dataset as it works.

Built for the **Meta Global AI Developer Hackathon** (Oct 26 – Nov 4, 2026), Indie Developer track.

![Overhead camera: SAM 3.1 finds the object; green shows where the arm can grasp](docs/images/overhead_view.jpg)

## Why

Anyone who builds hardware knows the moment: you're holding the iron and a wire, and the tool you need is just out of reach or buried on the bench. Makers, repair technicians and electronics labs lose small chunks of time to this all day. For people who can't easily reach across a bench, it's a real barrier.

Low-cost robot arms could help, but today they learn one task at a time from dozens of human demonstrations. Say-and-Grasp needs none. It understands the request, finds the tool and grasps it zero-shot. Each successful run then becomes a demonstration, so the arm can later be trained on data it collected itself.

## Meet Wiggle

The arm is called **Wiggle**. It naps folded up on the bench until you call its name. Then it wakes up: a sleepy twitch, a stretch with a big yawn (gripper wide open), a look around for you and a happy double snap of the gripper. While awake it sits up facing you, and it nods off again after a minute with nothing to do or when you say goodnight.

Its gripper is its mouth and its wrist is its head, so it can **wave**, **nod**, **shake its head**, **tilt its head** when puzzled (after a missed grab, say) and **dance**. Muse Spark picks a gesture when it fits: ask it to wave, thank it, or tell it the circuit works.

## How it works

You say *"Wiggle, pass me the side cutters"*. **Muse Voice Transcribe** turns it into text, and **Muse Spark** reads the request alongside a fresh photo of the bench. It works out which tool you mean and calls the robot's **`fetch`** tool. **SAM 3.1** finds the tool in the **overhead camera**; its outline gives a grasp point and jaw angle, which the calibration turns into a position on the table. The arm moves above the tool, and **SAM 3.1** looks again through the **wrist camera**. If the jaws are off-centre or not square to the tool, it **corrects and re-checks** until they're lined up. Then it **grasps and checks the grip**. Still holding the tool, it looks for **your open hand** with **SAM 3.1**, moves over it and **drops the tool into your palm**. If no hand is held out, it puts the tool on a handover spot and **Muse Spark checks** that it landed there. Either way, the whole run is saved as a **LeRobot dataset episode** labelled with how it went.

| Step | Meta model | What it does |
|---|---|---|
| Hear | **Muse Voice Transcribe** | Turns your spoken request into text |
| Understand | **Muse Spark** (vision + tool calling) | Looks at a photo of the bench, maps vague words to what's there ("the thing for cutting wire" → side cutters) and calls the robot's `fetch` tool |
| Find | **SAM 3.1**, overhead camera | Finds the tool from its name; the mask gives a grasp point and jaw angle, mapped to table coordinates |
| Align | **SAM 3.1**, wrist camera | Hovering above the tool, measures how far off-centre and how rotated it is relative to the jaws, corrects and re-checks. It only descends once the wrist camera confirms the jaws are lined up |
| Act | — | Top-down inverse kinematics on the SO-101, grip check |
| Hand over | **SAM 3.1**, overhead camera | Finds your outstretched hand (ignoring the robot's own gripper) and drops the tool into it |
| Verify | **Muse Spark** | When the tool goes on the handover spot instead, judges whether it landed there; that verdict labels the episode |

![Wrist-camera alignment before grasping](docs/images/wrist_alignment.jpg)

### Behaviour that makes it feel alive
- **Reaches most of the bench.** Straight down where it can, and leaning the gripper in or out near its base and at the far edge, so it grasps from 10 to about 36 cm away (up to 42 cm for some tools).
- **Never grabs blind.** It descends only when the wrist camera confirms the jaws are centred and square on the object.
- **Follows a moved object.** Nudge the tool while the arm is on its way and the wrist camera sees where it went; the arm follows it and re-aligns.
- **Recovers from a miss in place.** If the grip comes up empty, it rises, looks again and re-grabs without going back home.
- **Hands it to you.** It drops the tool into your open palm; if your hand is out of reach, it asks you to bring it closer.
- **Speaks naturally.** It says what it's doing as it starts ("On it, grabbing the side cutters") and confirms when it's done.
- **Honest about limits.** If something is out of reach, or it can't line up with it, it says so instead of trying anyway.
- **Remembers where things are.** *"Where's my multimeter?"* It points at it with the gripper, or says what it last did with it (*"I handed it to you ten minutes ago"*).
- **Tidies up.** *"Tidy up"* and Muse Spark looks at the bench, plans the order and puts each loose tool into the tray, one after another.

### Meta models per handover

A typical *"pass me the screwdriver"* uses **1** Muse Voice Transcribe call, **1–2** Muse Spark calls (deciding, and checking a bench drop) and **4–6** SAM 3.1 calls (finding the tool overhead, 2–3 wrist-camera alignment checks, finding your hand). No task-specific training, and nothing runs on a GPU of our own.

`sag-agent --brain-view` records a split-screen video of all of it as it happens: both cameras with SAM 3.1's masks, what you said, what Muse Spark decided and the verdict.

## The self-training loop

Robot arms usually learn a task from dozens of human demonstrations. Wiggle collects its own:

1. **Do the task zero-shot.** Muse Spark and SAM 3.1 handle every request without any training.
2. **Record everything.** Each `fetch` and `put_away` is saved at 15 fps as a [LeRobot](https://github.com/huggingface/lerobot) episode: both camera views, the arm's joints and the commands it was sent, and the task in words.
3. **Label it.** Every episode gets a verdict: Muse Spark checks a bench drop in the overhead view; a tool dropped into your hand counts as a success. Failures stay in the dataset but are marked as such.
4. **Train on the good ones.** `sag-train` picks the successful episodes and trains a policy (ACT) on Hugging Face Jobs. `sag-policy` runs it on the arm.

**No human demonstrations anywhere in the loop**: the zero-shot system is the teacher.

| Feature | Content |
|---|---|
| `observation.images.overhead` / `observation.images.wrist` | both camera views |
| `observation.state` / `action` | 5 arm joints + gripper, read and commanded |
| `task` | e.g. *"put the side cutters in the tray"* (language-conditioned, ready for SmolVLA) |
| `labels.jsonl` | success verdict and reason for each episode |

```bash
sag-dataset                 # episodes, success rate, per-object breakdown
sag-dataset --push          # upload to the Hugging Face Hub
sag-train --task tray       # what it would train on; add --go to train on HF Jobs
sag-policy --go             # run the trained policy on the arm
```

## Results

| | Trials | Success | Median time to hand over |
|---|---|---|---|
| Fetch into your hand (zero-shot) | 10 | 10 (100%) | 22.0 s |
| Tidy up into the tray (zero-shot) | — | — | — |
| ACT trained on Wiggle's own episodes | — | — | — |

Fetch trials: a screwdriver, tweezers (including lying on their side), a brush and a small block, placed 18–34 cm from the base in different directions and orientations, one request at a time with no help. Time runs from the typed request to the tool landing in the user's palm, including Muse Spark's decision. Of that, picking itself (overhead look, wrist alignment, grasp) took a median of 8.9 s, usually with just one or two wrist-camera checks.

## Hardware

- SO-101 follower arm with a parallel gripper (LeRobot)
- Overhead USB camera (Logitech 4K Stream, run at 1920×1080) looking straight down at the bench
- Wrist camera on the gripper (Innomaker U20CAM)
- A laptop with no GPU; all models run through Meta Model API
- A small tray for tidying up (any colour, walls ~4 cm)

## Setup

```bash
conda activate lerobot                # LeRobot env (arm control)
pip install -e .                      # installs the sag-* commands
export META_API_KEY=your-key          # dev.meta.ai → API keys
sudo apt install -y libportaudio2 espeak-ng   # microphone, fallback voice
python -m piper.download_voices en_US-lessac-medium --download-dir assets/voices   # Wiggle's voice

# SO-101 model for kinematics (once)
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git /tmp/soarm
cp -r /tmp/soarm/Simulation/SO101 assets/
```

One-time calibration, in this order (each takes a few minutes):

```bash
sag-calibrate --tcp                                  # where your gripper actually grasps
sag-calibrate --camera 4 --marker "blue block" --points 10   # overhead camera ↔ arm (near, middle and far touches)
sag-calibrate --refit --marker-height 11             # if you touched the TOP of an 11 mm marker
sag-pick --save-home                                 # folded rest pose, out of the camera's view
sag-pick --save-clear                                # rest pose with the wrist lifted off the base
sag-calibrate --wrist --wrist-camera 2 --marker "blue block"   # wrist-camera alignment
sag-spots set handover 1260,840                      # fallback spot if no hand is held out (overhead pixel)
sag-pick --show-reach                                # check the reachable zone (outputs/reach.png)
sag-wiggle --save-awake                                # Wiggle's awake pose: sitting up, facing you, out of the camera's view
```

## Use

```bash
sag-agent --hands-free        # just talk: "Wiggle!" wakes it, then "pass me the pliers"
sag-agent                     # push-to-talk: Enter, speak, Enter
sag-agent --text              # type requests instead
sag-agent --text --push       # …and upload the dataset to the Hub when you quit
sag-agent --hands-free --brain-view   # also record what it sees and thinks (outputs/brainview/)
```

Lower-level tools: `sag-segment` (run SAM 3.1 on a camera or image), `sag-pick` (pick one named object without the assistant; `--offer` hands it to you), `sag-wiggle wake wave dance sleep` (play Wiggle's gestures).

## Repo layout

```
src/say_and_grasp/
  agent.py         Muse Spark assistant (Wiggle's personality): tools, one-call fetch, handover verification
  character.py     Wiggle's wake word, wake-up and sleep animations, gestures
  skills.py        look / pick / place / hand over: wrist-camera alignment, follow, retry, drop into your hand
  sam.py           SAM 3.1 via Meta Model API: prompt → masks → grasp targets
  voice.py         push-to-talk or hands-free listening, Muse Voice Transcribe, spoken replies
  recorder.py      every fetch / put-away → LeRobot dataset episode
  memory.py        where each tool was last seen, picked, handed over or put away
  brainview.py     split-screen video of the cameras, SAM masks and Muse Spark's decisions
  events.py        how skills and the assistant report what they're doing
  kinematics.py    IK on the SO-101 URDF (placo): straight down, or leaning in/out at the edges of reach
  arm.py           smooth, gravity-compensated arm and gripper control (LeRobot)
  calibration.py   overhead pixel ↔ table mapping (with a reach correction), table plane, reach limits
  camera.py        single shots and background camera streams
  spots.py         named bench spots (handover)
  cli/             sag-agent, sag-pick, sag-wiggle, sag-calibrate, sag-segment, sag-dataset, sag-train
config/            per-setup calibration (camera, gripper, wrist camera, home pose)
assets/SO101/      SO-101 URDF + meshes (fetched, not committed)
assets/voices/     Piper voice for Wiggle (downloaded, not committed)
data/              recorded episodes (pushed to the Hub, not committed)
docs/images/       README images
```

## What existed before the hackathon

To keep the judging clear: the SO-101 arm hardware and its LeRobot setup (motor calibration, teleoperation) existed before the event. Everything in this repository was written during the hackathon.

## License

[MIT](LICENSE)
