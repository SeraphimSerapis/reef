"""notify-gate: should a Home Assistant notification interrupt anyone, and how?"""

from __future__ import annotations

import random

from reef.packs.base import Pack, QuestionSpec, clock, maybe, pick

NOTIFICATIONS = [
    ("washing_machine", "Laundry", "The washing machine has finished."),
    ("dishwasher", "Kitchen", "Dishwasher done. Door can be opened."),
    ("water_leak", "Leak sensor", "Water detected under the {room} sink!"),
    ("smoke", "Smoke alarm", "Smoke detected in the {room}."),
    ("co", "CO alarm", "Carbon monoxide level {ppm} ppm in the {room}."),
    ("door_open", "Doors", "The {door} has been open for {mins} minutes."),
    ("door_unlocked", "Locks", "The {door} is unlocked."),
    ("window_rain", "Weather", "Rain expected in 20 minutes and the {room} window is open."),
    ("low_battery", "Batteries", "{device} battery at {pct}%."),
    ("freezer_temp", "Freezer", "Freezer temperature is {temp} °C."),
    ("update", "Updates", "Home Assistant Core {ver} is available."),
    ("device_offline", "Devices", "{device} has been unavailable for {mins} minutes."),
    ("energy_price", "Energy", "Electricity price is negative for the next hour."),
    ("solar_export", "Solar", "Exporting {kw} kW to the grid."),
    ("car_charged", "Car", "The car is charged to {pct}%."),
    ("doorbell", "Doorbell", "Someone rang the doorbell."),
    ("package", "Deliveries", "A package was delivered."),
    ("alarm_triggered", "Alarm", "Alarm triggered: motion in the {room} while armed."),
    ("internet_down", "Network", "Internet connection is down (failover to LTE)."),
    ("plant", "Plants", "The {plant} needs water."),
    ("bin_day", "Reminders", "Bin collection tomorrow: {bin}."),
    ("garage_open_night", "Garage", "The garage door is open and it is after sunset."),
    ("server_alert", "Homelab", "NAS: pool tank degraded."),
    ("printer", "Printer", "Printer toner low."),
    ("sump_pump", "Basement", "Sump pump running continuously for {mins} minutes."),
]

ROOMS = ["kitchen", "bathroom", "basement", "bedroom", "living room", "laundry room"]
DOORS = ["front door", "back door", "patio door", "garage side door"]
DEVICES = ["Hallway motion sensor", "Front door lock", "Smoke detector kitchen", "Thermostat", "Zigbee coordinator",
           "Aqara leak sensor", "Garage door tilt sensor"]


def generate(rng: random.Random) -> dict:
    kind, source, template = rng.choice(NOTIFICATIONS)
    msg = template.format(room=rng.choice(ROOMS), door=rng.choice(DOORS), mins=rng.choice([2, 10, 30, 90, 240]),
                          device=rng.choice(DEVICES), pct=rng.choice([1, 5, 10, 15, 25, 80, 100]),
                          temp=rng.choice([-21, -18, -12, -6, 2]), ver="2026.10.1", kw=round(rng.uniform(0.5, 9), 1),
                          ppm=rng.choice([9, 35, 70, 150, 400]), plant="monstera",
                          bin=rng.choice(["recycling", "general waste", "garden waste"]))
    when, hour = clock(rng, night_bias=0.35)
    household = {
        "people_home": rng.choice([[], ["Alex"], ["Alex", "Sam"], ["Sam", "kids"], ["Alex", "Sam", "kids"]]),
        "sleeping": hour >= 23 or hour < 7 if rng.random() < 0.85 else rng.random() < 0.3,
        "do_not_disturb": maybe(rng, 0.15, True, False),
        "guests_over": maybe(rng, 0.1, True, False),
        "on_vacation": maybe(rng, 0.1, True, False),
    }
    if household["on_vacation"]:
        household["people_home"] = []
    return {
        "source": "home-assistant",
        "notification": {"from": source, "kind": kind, "message": msg,
                         "requested_priority": pick(rng, {"normal": 0.7, "high": 0.2, "critical": 0.1})},
        "local_time": when,
        "household": household,
        "same_notification_last_24h": rng.choice([0, 0, 0, 1, 3, 12]),
    }


PACK = Pack(
    name="notify-gate",
    description="Gate Home Assistant notifications: drop, digest, push, or break through do-not-disturb.",
    generate=generate,
    tags=["home-assistant", "notifications"],
    teacher_notes="""\
Context: Home Assistant wants to send this notification to the household's phones. requested_priority is what the automation asked for; automations often over-ask. same_notification_last_24h counts repeats.
House policy:
- Life safety always breaks through, even while sleeping or on do-not-disturb: smoke, CO, water leak, freezing pipes, a door or window opened while nobody is home or everyone is asleep, high CO2 in a bedroom.
- Security events while on vacation or away are worth a push; the same event while people are home and awake usually is not.
- Chores and status (laundry done, bins tomorrow, plants, software updates, energy reports, low batteries above about 10%) go to the daily digest, or get dropped if they repeat.
- Do not wake sleeping people unless it is a safety issue. Respect do-not-disturb except for safety.
- Repeats of the same notification are less important than the first one.""",
    questions=[
        QuestionSpec("channel", "choice", [
            "How should this notification be delivered?",
            "Pick the delivery channel for this notification.",
        ], {"drop": "don't deliver it at all", "digest": "include it in the daily summary",
            "push": "normal push notification", "critical": "loud alert that breaks through do-not-disturb"}),
        QuestionSpec("wake_up", "noul", [
            "Is this worth waking someone up for?",
            "Should this wake a sleeping person?",
        ]),
        QuestionSpec("urgency", "score", [
            "How time-sensitive is this?",
            "How urgent is this notification?",
        ], ["whenever", "today", "within the hour", "within minutes", "immediately, safety issue"]),
        QuestionSpec("safety", "noul", [
            "Is this a safety issue for people or the house?",
            "Could ignoring this put people or property at risk?",
        ]),
    ],
)
