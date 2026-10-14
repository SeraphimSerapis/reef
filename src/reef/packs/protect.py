"""protect-triage: decide what to do about a UniFi Protect event, from its metadata.

Field names follow Protect's own event model (``type``, ``smartDetectTypes``, ``score``, audio
``alrm*`` types, doorbell ``ring``, camera health events) so a Home Assistant automation or a
small webhook bridge can pass events through almost verbatim. Protect has already done the
vision on the camera / NVR; Reef decides whether a human should care.

The big real-world problem this targets: Protect is *chatty*. A typical home sees tens of
thousands of ``alrmSpeak`` audio events and motion events per month, and a few events that
really matter (smoke, CO, glass break, someone in the backyard at 3am).
"""

from __future__ import annotations

import random

from reef.packs.base import Pack, QuestionSpec, clock, maybe, pick

CAMERAS = {
    # name: (model, typical smart detections)
    "Front Door": ("G4 Doorbell Pro", ["person", "package", "face"]),
    "Driveway": ("G5 Bullet", ["vehicle", "person", "licensePlate"]),
    "Backyard": ("G5 Flex", ["person", "animal"]),
    "Garden PTZ": ("G5 PTZ", ["person", "animal", "vehicle"]),
    "Garage": ("G4 Instant", ["person", "vehicle"]),
    "Side Path": ("G3 Flex", ["person", "animal"]),
    "Nursery": ("G4 Instant", ["person"]),
    "Living Room": ("AI 360", ["person", "animal"]),
    "Street": ("AI Bullet", ["vehicle", "person", "licensePlate"]),
}
INDOOR = {"Nursery", "Living Room", "Garage"}
RESIDENTS = ["Alex", "Sam", "Jordan", "Mia"]
KNOWN = ["Grandma", "Neighbour Tom", "Cleaner Ana", "Dog walker"]
PLATES = {"HB AB 123": "family car", "B XY 9001": "Sam's car", "M KL 4410": "Grandma"}

AUDIO = {"alrmSpeak": 8, "alrmBark": 1.5, "alrmCarHorn": 0.6, "alrmSiren": 0.5, "alrmBabyCry": 0.8,
         "alrmSmoke": 0.35, "alrmCmonx": 0.15, "alrmGlassBreak": 0.3, "alrmBurglar": 0.2}
HEALTH = {"poorConnection": 2, "cameraDisconnected": 1, "deviceDisconnected": 0.6, "lowMemory": 0.4,
          "fogDetection": 0.4, "resolutionLowered": 1.2, "storageFull": 0.2, "deviceConnected": 0.8}


def _smart(rng: random.Random, cam: str) -> dict:
    model, usual = CAMERAS[cam]
    primary = rng.choice(usual)
    if primary == "face":
        primary = "person"
    types = {primary}
    if primary == "person" and rng.random() < 0.3:
        types.add("face")
    if rng.random() < 0.15:
        types.add(rng.choice(["person", "vehicle", "animal"]))
    ev: dict = {
        "type": pick(rng, {"smartDetectZone": 6, "smartDetectLine": 1, "smartDetectLoiterZone": 0.7}),
        "smartDetectTypes": sorted(types),
        "score": rng.randint(50, 98),
        "zone": maybe(rng, 0.6, rng.choice(["Entrance", "Lawn", "Gate", "Parking", "Porch", "Fence", "Pool"])),
    }
    if "person" in types:
        r = rng.random()
        if "face" in types or r < 0.35:
            q = rng.random()
            if q < 0.45:
                ev["face"] = {"name": rng.choice(RESIDENTS), "group": "household", "confidence": rng.randint(70, 98)}
            elif q < 0.65:
                ev["face"] = {"name": rng.choice(KNOWN), "group": "known", "confidence": rng.randint(65, 95)}
            else:
                ev["face"] = {"name": None, "group": "unknown"}
        if rng.random() < 0.25:
            ev["direction"] = rng.choice(["towards camera", "away from camera", "left to right", "right to left"])
    if "vehicle" in types and "licensePlate" in CAMERAS[cam][1] and rng.random() < 0.6:
        if rng.random() < 0.5:
            plate = rng.choice(list(PLATES))
            ev["licensePlate"] = {"plate": plate, "known_as": PLATES[plate]}
        else:
            ev["licensePlate"] = {"plate": f"{rng.choice(['K', 'HH', 'F', 'D'])} {rng.choice('ABCDEFGH')}"
                                           f"{rng.choice('KLMNOP')} {rng.randint(10, 9999)}", "known_as": None}
    if "package" in types:
        ev["smartDetectTypes"] = ["package"]
    return ev


def generate(rng: random.Random) -> dict:
    kind = pick(rng, {"smart": 5, "audio": 2.5, "ring": 0.8, "motion": 1.5, "health": 1.2})
    cam = rng.choice(list(CAMERAS))
    model = CAMERAS[cam][0]
    when, hour = clock(rng, night_bias=0.35)
    night = hour >= 22 or hour < 6

    if kind == "smart":
        ev = _smart(rng, cam)
    elif kind == "audio":
        audio = pick(rng, AUDIO)
        ev = {"type": "smartAudioDetect", "smartDetectTypes": [audio], "score": rng.randint(40, 95)}
    elif kind == "ring":
        cam, model = "Front Door", "G4 Doorbell Pro"
        ev = {"type": "ring"}
        if rng.random() < 0.6:
            ev["also_seen"] = _smart(rng, cam)["smartDetectTypes"]
    elif kind == "motion":
        ev = {"type": "motion", "score": rng.randint(5, 90)}
    else:
        ev = {"type": pick(rng, HEALTH)}
        if ev["type"] == "storageFull":
            cam, model = "NVR", "UNVR"
    ev = {"camera": cam, "model": model, **ev, "start": when}
    if kind != "health":
        ev["duration_s"] = rng.choice([2, 5, 10, 20, 45, 90, 300, 900])
    if kind not in ("ring", "health") and rng.random() < 0.7:
        ev["same_camera_last_hour"] = pick(rng, {0: 3, 1: 2, 3: 1.5, 8: 1, 25: 0.6, 80: 0.3})

    household = {
        "people_home": rng.sample(RESIDENTS, k=rng.choice([0, 0, 1, 2, 2, 3, 4])),
        "alarm": pick(rng, {"disarmed": 0.55, "armed_home": 0.2, "armed_away": 0.25}),
        "quiet_hours": night or rng.random() < 0.08,
    }
    if household["alarm"] == "armed_away":
        household["people_home"] = []
    if rng.random() < 0.2:
        household["expecting_delivery"] = True
    if cam == "Nursery" and rng.random() < 0.7:
        household["baby_asleep"] = True
    return {"source": "unifi-protect", "event": ev, "household": household}


PACK = Pack(
    name="protect-triage",
    description="Decide what to do about a UniFi Protect event (smart detection, audio, doorbell, camera health).",
    generate=generate,
    tags=["unifi", "protect", "nvr", "home-assistant", "security"],
    teacher_notes="""\
Glossary for UniFi Protect events:
- type: smartDetectZone = object detected in a zone; smartDetectLine = object crossed a line; smartDetectLoiterZone = object lingered in a zone; smartAudioDetect = sound detected; ring = doorbell pressed; motion = plain pixel motion (often wind, shadows, insects).
- Audio types: alrmSpeak = people talking (harmless chatter, very common); alrmBark = dog barking; alrmCarHorn = car horn; alrmSiren = emergency vehicle siren passing; alrmBabyCry = baby crying; alrmSmoke = smoke alarm beeping; alrmCmonx = carbon-monoxide alarm beeping; alrmGlassBreak = glass breaking; alrmBurglar = burglar alarm sounding.
- score = detection confidence 0-100. face.group: household = lives here, known = trusted visitor, unknown = stranger. same_camera_last_hour = how many events this camera already produced in the last hour (high = noisy, repeat).
- Camera health types (poorConnection, cameraDisconnected, resolutionLowered, lowMemory, fogDetection, storageFull) are about the equipment, not about people.
House policy:
- Smoke, CO and glass break are emergencies: always notify urgently, even during quiet hours, even if people are home.
- A burglar alarm sounding, or an unknown person lingering or loitering at night while the alarm is armed, is a likely intrusion: notify urgently (or sound the alarm when nobody is home).
- Household members, known visitors, chatter, barking, passing cars and plain motion are not worth a notification. Repeats are less interesting than the first event.
- A doorbell ring is worth a normal notification. A delivery is worth a normal notification, especially when one is expected.
- A baby crying in the nursery while the baby is asleep deserves a notification.
- Camera disconnects and full storage deserve a non-urgent heads-up; brief poor connections do not.""",
    questions=[
        QuestionSpec("notify", "noul", [
            "Should the household get a phone notification for this event?",
            "Is this Protect event worth a push notification?",
            "Should someone be told about this right now?",
        ]),
        QuestionSpec("urgency", "score", [
            "How urgent is this event?",
            "Rate how quickly a human needs to look at this.",
        ], ["ignore", "whenever", "today", "soon", "right now"]),
        QuestionSpec("action", "choice", [
            "What should the system do?",
            "Pick the right response to this event.",
        ], {"ignore": "drop the event", "log": "keep the clip, no notification",
            "notify": "push notification with snapshot", "notify_urgent": "loud notification that breaks through do-not-disturb",
            "alarm": "turn on lights and sound the siren"}),
        QuestionSpec("category", "choice", [
            "What best describes this event?",
            "Classify this event.",
        ], {"household": "someone who lives here or their car", "visitor": "a known visitor or someone at the door",
            "delivery": "a courier or a parcel", "passerby": "someone or something just passing by",
            "animal": "a pet or wildlife", "suspicious": "unknown activity that looks out of place",
            "safety": "smoke, CO, glass break or a similar emergency", "noise": "chatter, barking or other harmless sound",
            "camera_health": "a problem with the camera or recorder itself"}),
        QuestionSpec("is_emergency", "noul", [
            "Is this a possible life-safety emergency?",
            "Could this be a fire, CO or break-in emergency?",
        ]),
    ],
)
