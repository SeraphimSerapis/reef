"""frigate-triage: decide what to do about a Frigate NVR event, from its metadata.

Frigate has already done the vision (on its own detector). Reef reasons over the event's
metadata (label, score, zones, time, recognised face / plate, whether the object is stationary)
plus household context, and optionally Frigate's GenAI text description of the clip.
"""

from __future__ import annotations

import random

from reef.packs.base import Pack, QuestionSpec, clock, maybe, pick

CAMERAS = {
    "front_door": ["porch", "doorstep", "walkway"],
    "driveway": ["driveway", "garage_door", "street"],
    "backyard": ["patio", "lawn", "shed", "fence_line"],
    "side_gate": ["gate", "bins"],
    "garage": ["inside_garage", "workbench"],
    "street": ["street", "sidewalk"],
}
RESIDENTS = ["Alex", "Sam", "Jordan", "Mia"]
KNOWN = ["Grandma", "Neighbour Tom", "Cleaner Ana"]
PLATES_KNOWN = {"HB-AB 123": "family car", "B-XY 9001": "Sam's car"}

DESCRIPTIONS = {
    "person": [
        "A delivery driver in a uniform carries a parcel to the door, leaves it and walks back to a van.",
        "A person in a dark hoodie walks slowly along the fence, looking into the windows.",
        "A person tries the handle of the car door, then walks off quickly.",
        "Two children walk past on the sidewalk.",
        "A resident arrives home carrying groceries and unlocks the door.",
        "A person rings the doorbell and waits holding a clipboard.",
        "A person climbs over the side gate.",
        "A jogger runs past on the street.",
        "A person shines a flashlight into the garage window.",
        "A gardener mows the lawn.",
    ],
    "car": [
        "A car pulls into the driveway and parks.",
        "A car slows down in front of the house, idles for a minute and drives off.",
        "A car drives past on the street.",
        "A car is parked across the driveway, blocking it.",
    ],
    "dog": ["A dog sniffs around the lawn.", "The family dog lies on the patio."],
    "cat": ["A cat walks along the fence.", "A cat sits on the car bonnet."],
    "package": ["A parcel sits on the doorstep.", "A parcel has been on the porch for a while."],
    "bicycle": ["A bicycle leans against the garage.", "Someone rides a bicycle up the driveway."],
    "bird": ["A bird lands on the camera housing."],
    "fox": ["A fox crosses the lawn at night."],
}


def generate(rng: random.Random) -> dict:
    cam = rng.choice(list(CAMERAS))
    label = pick(rng, {"person": 5, "car": 2.5, "dog": 0.8, "cat": 1, "package": 1, "bicycle": 0.5, "bird": 0.5,
                       "fox": 0.3})
    if cam == "street" and label == "package":
        label = "car"
    when, hour = clock(rng, night_bias=0.35)
    night = hour >= 22 or hour < 6
    zones = rng.sample(CAMERAS[cam], k=rng.randint(0, min(2, len(CAMERAS[cam]))))
    top = round(rng.uniform(0.55, 0.97), 2)
    ev = {
        "camera": cam,
        "label": label,
        "top_score": top,
        "entered_zones": zones,
        "start": when,
        "duration_s": rng.choice([3, 8, 15, 30, 60, 180, 600, 1800]),
        "stationary": rng.random() < (0.5 if label in ("car", "package") else 0.15),
        "false_positive_hint": maybe(rng, 0.08, rng.choice(["low light", "rain on lens", "spider web", "headlights"])),
    }
    if label == "person":
        r = rng.random()
        if r < 0.3:
            ev["sub_label"] = {"face": rng.choice(RESIDENTS), "score": round(rng.uniform(0.7, 0.98), 2)}
        elif r < 0.4:
            ev["sub_label"] = {"face": rng.choice(KNOWN), "score": round(rng.uniform(0.7, 0.95), 2)}
        elif r < 0.55:
            ev["sub_label"] = {"face": "unknown"}
    if label == "car" and rng.random() < 0.5:
        if rng.random() < 0.5:
            plate = rng.choice(list(PLATES_KNOWN))
            ev["recognized_plate"] = {"plate": plate, "known_as": PLATES_KNOWN[plate]}
        else:
            ev["recognized_plate"] = {"plate": f"{rng.choice(['M', 'K', 'HH', 'F'])}-{rng.choice('ABCDEFGH')}"
                                      f"{rng.choice('KLMNOP')} {rng.randint(10, 9999)}", "known_as": None}
    if rng.random() < 0.45:
        ev["genai_description"] = rng.choice(DESCRIPTIONS[label])
    household = {
        "anyone_home": maybe(rng, 0.65, True, False),
        "alarm": pick(rng, {"disarmed": 0.55, "armed_home": 0.2, "armed_away": 0.25}),
        "expecting_delivery": maybe(rng, 0.2, True, False),
        "night_mode": night,
    }
    if household["alarm"] == "armed_away":
        household["anyone_home"] = False
    return {"source": "frigate", "event": ev, "household": household}


PACK = Pack(
    name="frigate-triage",
    description="Decide what to do about a Frigate NVR event from its metadata and household context.",
    generate=generate,
    tags=["frigate", "nvr", "home-assistant", "security"],
    teacher_notes="""\
Glossary for Frigate events: label = detected object; top_score = detector confidence 0-1; entered_zones = zones the object entered (empty = only seen in the frame); stationary = object is not moving (parked car, parcel); sub_label.face = face recognition result (a resident, a known visitor, or unknown); recognized_plate.known_as = a known car; genai_description = an LLM's description of the clip (trust it more than the label); false_positive_hint = conditions that often cause misdetections.
House policy:
- Residents and their cars are not worth a notification. Known visitors, deliveries and parcels are worth a normal notification, especially when a delivery is expected.
- Passers-by on the street, animals and low-confidence detections with a false-positive hint are noise.
- Strangers lingering, looking into windows, trying door handles, climbing gates or fences, or moving around at night are suspicious; with the alarm armed away and nobody home, that warrants an urgent notification or the siren.
- A stationary car blocking the driveway is worth a normal notification.""",
    questions=[
        QuestionSpec("notify", "noul", [
            "Should the household get a phone notification for this event?",
            "Is this event worth a push notification?",
            "Should anyone be told about this right now?",
        ]),
        QuestionSpec("threat", "score", [
            "How concerning is this event from a security point of view?",
            "Rate the threat level of this event.",
        ], ["nothing to see", "unremarkable", "worth a look later", "suspicious", "possible intrusion or theft"]),
        QuestionSpec("action", "choice", [
            "What should the system do?",
            "Pick the right response to this event.",
        ], {"ignore": "drop the event", "log": "keep the clip, no notification",
            "notify": "push notification with snapshot", "notify_urgent": "loud notification that breaks through do-not-disturb",
            "alarm": "turn on lights and sound the siren"}),
        QuestionSpec("category", "choice", [
            "What best describes this event?",
            "Classify this event.",
        ], {"resident": "someone who lives here or their car", "visitor": "a known or expected visitor",
            "delivery": "a courier or a parcel", "passerby": "someone or something just passing by",
            "animal": "a pet or wildlife", "suspicious": "unknown activity that looks out of place",
            "false_alarm": "probably a misdetection"}),
        QuestionSpec("is_false_positive", "noul", [
            "Is this likely a misdetection?",
            "Is the detection probably wrong?",
        ]),
    ],
)
