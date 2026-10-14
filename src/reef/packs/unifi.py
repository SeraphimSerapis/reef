"""unifi-network: triage UniFi Network events (WAN, devices, clients, IPS, admin access).

Event keys and parameter names mirror what the UniFi Network application stores for its system
log (``ISP_HIGH_LATENCY_2``, ``NETWORK_WAN_FAILED_TEMPORARY_2``, ``DEVICE_UNREACHABLE``,
``LOG_SWITCH_PORT_LINK_FLAP``, ``ADMIN_ACCESS`` ...), flattened to the fields that matter. The
typical home controller logs thousands of client connect / disconnect / roam events per week and
a handful that deserve a human: a WAN going down, an AP dropping off, an admin login from an
unfamiliar IP, an IPS hit from an IoT device.
"""

from __future__ import annotations

import random

from reef.packs.base import Pack, QuestionSpec, clock, maybe, pick

DEVICES = {
    "AP Upstairs": "U6-LR", "AP Downstairs": "U6-Pro", "AP Garden": "U6-Mesh", "AP Guest": "UAP-AC-IW",
    "Switch Basement": "USW-Pro-HD-24-PoE", "Switch Office": "USW-Lite-8-PoE", "Switch Attic": "USW-Flex-Mini",
    "Gateway": "UDM-Pro",
}
WLANS = {"Home": "trusted", "Home_IoT": "iot", "Home_Guest": "guest"}
CLIENTS = {
    "trusted": ["Alex iPhone", "Sam MacBook", "Mia iPad", "Office NAS", "Desktop-PC", "Jordan Pixel"],
    "iot": ["Netatmo Weather", "Hue Bridge", "Robot Vacuum", "Shelly Plug 3", "Smart TV", "Wallbox", "Printer",
            "e-ink Display", "Tesla", "Solar Inverter"],
    "guest": ["Galaxy-S24", "iPhone", "Unknown laptop"],
}
UNKNOWN_MAC = ["02:5e:a1:7c:90:3b", "3a:f1:0c:22:9e:47", "b2:7d:13:aa:01:5c"]  # locally administered, made up
COUNTRIES = ["DE", "NL", "US", "CN", "RU", "BR", "FR", "SG"]
IPS = [
    ("ET SCAN Suspicious inbound to mySQL port 3306", "scan", "inbound"),
    ("ET POLICY Outgoing Basic Auth Base64 HTTP Password detected unencrypted", "policy", "outbound"),
    ("ET MALWARE Possible Mirai Variant CnC Beacon", "malware", "outbound"),
    ("ET DROP Spamhaus DROP Listed Traffic Inbound", "reputation", "inbound"),
    ("ET TOR Known Tor Exit Node Traffic", "tor", "outbound"),
    ("ET COINMINER Stratum Login Attempt", "coinminer", "outbound"),
    ("ET P2P BitTorrent DHT ping request", "p2p", "outbound"),
    ("ET SCAN NMAP OS Detection Probe", "scan", "inbound"),
]

CHURN = ["CLIENT_CONNECTED_WIRELESS_2", "CLIENT_DISCONNECTED_WIRELESS_2", "CLIENT_ROAMED_2",
         "CLIENT_CONNECTED_WIRED_2", "CLIENT_DISCONNECTED_WIRED_2"]
KEYS = {
    "churn": 5, "NEW_WIFI_CLIENT_CONNECTED": 1.2, "ISP_HIGH_LATENCY_2": 1, "ISP_PACKET_LOSS_2": 0.8,
    "NETWORK_WAN_FAILED_TEMPORARY_2": 0.8, "NETWORK_WAN_FAILED_2": 0.6, "NETWORK_WAN_FAILED_MULTIPLE_TIMES_2": 0.5,
    "NETWORK_WAN_RESTORED_2": 0.5, "DEVICE_UNREACHABLE": 0.9, "DEVICE_RECONNECTED": 0.5,
    "LOG_SWITCH_PORT_LINK_FLAP": 0.6, "ADMIN_ACCESS": 0.9, "IPS_THREAT_DETECTED": 1.4, "FIRMWARE_UPDATED_V2": 0.5,
    "BULK_FIRMWARE_UPDATE_AVAILABLE": 0.4, "ROGUE_AP_DETECTED": 0.5, "POE_OVERLOAD": 0.3, "DEVICE_OVERHEATING": 0.25,
}


def _client(rng: random.Random) -> dict:
    wlan = rng.choice(list(WLANS))
    seg = WLANS[wlan]
    return {"name": rng.choice(CLIENTS[seg]), "network": wlan, "segment": seg}


def generate(rng: random.Random) -> dict:
    key = pick(rng, KEYS)
    if key == "churn":
        key = rng.choice(CHURN)
    when, hour = clock(rng, night_bias=0.25)
    ev: dict = {"key": key, "time": when}
    wan = rng.choice(["WAN1", "WAN2"])
    dev = rng.choice(list(DEVICES))

    if key.startswith("CLIENT_"):
        ev["client"] = _client(rng)
        if "WIRED" in key:
            ev["client"]["network"] = {"trusted": "Default", "iot": "IoT VLAN", "guest": "Guest VLAN"}[
                ev["client"]["segment"]]
        if "ROAMED" in key:
            a, b = rng.sample([d for d in DEVICES if d.startswith("AP")], 2)
            ev["from_ap"], ev["to_ap"] = a, b
        ev["client_events_last_hour"] = pick(rng, {1: 3, 4: 2, 12: 1, 40: 0.5})
    elif key == "NEW_WIFI_CLIENT_CONNECTED":
        c = _client(rng)
        if rng.random() < 0.5:
            c["name"] = rng.choice(UNKNOWN_MAC)
            c["hostname"] = maybe(rng, 0.4, rng.choice(["espressif", "android-7f3a", "DESKTOP-K2J9", "raspberrypi"]))
        ev["client"] = c
        ev["ap"] = rng.choice([d for d in DEVICES if d.startswith("AP")])
    elif key.startswith("ISP_"):
        ev["wan"] = wan
        ev["latency_ms" if "LATENCY" in key else "packet_loss_pct"] = (
            rng.choice([43, 80, 150, 400, 1200]) if "LATENCY" in key else rng.choice([2, 5, 12, 30, 60]))
        ev["duration_min"] = rng.choice([1, 5, 15, 60, 240])
    elif key.startswith("NETWORK_WAN_"):
        ev["wan"] = wan
        ev["failover_active"] = rng.random() < 0.6 if "RESTORED" not in key else False
        ev["other_wan_up"] = rng.random() < 0.75
        if "MULTIPLE" in key:
            ev["count"] = rng.choice([2, 3, 6, 15])
        if "RESTORED" in key:
            ev["outage_min"] = rng.choice([1, 4, 20, 90, 600])
    elif key in ("DEVICE_UNREACHABLE", "DEVICE_RECONNECTED"):
        dev = rng.choice([d for d in DEVICES if d != "Gateway"])
        ev["device"] = {"name": dev, "model": DEVICES[dev]}
        ev["uplink"] = maybe(rng, 0.7, {"name": "Switch Basement", "port": rng.randint(1, 24)})
        ev["clients_affected"] = rng.choice([0, 2, 8, 25])
        if key == "DEVICE_RECONNECTED":
            ev["offline_min"] = rng.choice([1, 3, 15, 120])
    elif key == "LOG_SWITCH_PORT_LINK_FLAP":
        sw = rng.choice([d for d in DEVICES if d.startswith("Switch")])
        ev["device"] = {"name": sw, "model": DEVICES[sw]}
        ev["port"] = rng.randint(1, 24)
        ev["port_name"] = maybe(rng, 0.5, rng.choice(["AP Upstairs", "NAS", "Camera Garage", "Uplink", "Desk"]))
        ev["count"] = rng.choice([3, 10, 40, 96, 300])
    elif key == "ADMIN_ACCESS":
        local = rng.random() < 0.55
        ev["admin"] = rng.choice(["Owner", "Owner", "Alex", "api-key: home-assistant", "Installer"])
        ev["ip"] = rng.choice(["192.168.1.23", "10.0.0.5"]) if local else rng.choice(
            ["198.51.100.23", "203.0.113.77", "192.0.2.146", "203.0.113.50"])  # RFC 5737 documentation ranges
        ev["remote"] = not local
        if not local:
            ev["country"] = pick(rng, {"DE": 4, "NL": 0.6, "US": 0.6, "RU": 0.4, "CN": 0.4, "BR": 0.3})
            ev["known_location"] = rng.random() < 0.6
        ev["platform"] = rng.choice(["web", "mobile", "api", "ssh"])
        ev["mfa"] = rng.random() < 0.7
    elif key == "IPS_THREAT_DETECTED":
        sig, cat, direction = rng.choice(IPS)
        ev["signature"] = sig
        ev["category"] = cat
        ev["direction"] = direction
        ev["action"] = rng.choice(["blocked", "blocked", "alert"])
        ev["client" if direction == "outbound" else "target"] = _client(rng) if direction == "outbound" else {
            "port": rng.choice([22, 80, 443, 3306, 8123, 32400])}
        ev["remote_country"] = rng.choice(COUNTRIES)
        ev["hits_24h"] = rng.choice([1, 3, 20, 400])
    elif key.startswith("FIRMWARE") or key.startswith("BULK"):
        ev["device"] = {"name": dev, "model": DEVICES[dev]}
        if key.startswith("BULK"):
            ev["devices"] = rng.randint(2, 7)
            ev["security_fix"] = rng.random() < 0.3
    elif key == "ROGUE_AP_DETECTED":
        ev["ssid"] = rng.choice(["Home", "FRITZ!Box 7590 XY", "DIRECT-roku-123", "Free WiFi", "Home_Guest"])
        ev["rssi_dbm"] = rng.choice([-85, -70, -55, -40])
        ev["seen_by"] = rng.choice([d for d in DEVICES if d.startswith("AP")])
    elif key == "POE_OVERLOAD":
        sw = rng.choice([d for d in DEVICES if d.startswith("Switch")])
        ev["device"] = {"name": sw, "model": DEVICES[sw]}
        ev["budget_used_pct"] = rng.choice([95, 101, 115])
        ev["port_shut_down"] = rng.random() < 0.4
    elif key == "DEVICE_OVERHEATING":
        ev["device"] = {"name": dev, "model": DEVICES[dev]}
        ev["temp_c"] = rng.choice([68, 75, 85, 95])

    context = {
        "owner_home": rng.random() < 0.6,
        "quiet_hours": hour >= 23 or hour < 7,
        "maintenance_window": rng.random() < 0.08,
    }
    return {"source": "unifi-network", "event": ev, "context": context}


PACK = Pack(
    name="unifi-network",
    description="Triage UniFi Network events: WAN health, device outages, clients, IPS hits and admin logins.",
    generate=generate,
    tags=["unifi", "network", "security", "homelab"],
    teacher_notes="""\
Glossary for UniFi Network events:
- CLIENT_CONNECTED/DISCONNECTED/ROAMED = routine client churn; phones and IoT devices do this all day. client_events_last_hour high = flapping client.
- NEW_WIFI_CLIENT_CONNECTED = a device never seen before joined; a bare MAC address or generic hostname on the trusted network deserves a look, on the guest network it is normal.
- ISP_HIGH_LATENCY / ISP_PACKET_LOSS = the internet connection is degraded. NETWORK_WAN_FAILED(_TEMPORARY) = a WAN link went down; if failover_active / other_wan_up, the home is still online. WAN_FAILED_MULTIPLE_TIMES = a flapping uplink.
- DEVICE_UNREACHABLE = a UniFi AP or switch went offline (clients_affected matter). LOG_SWITCH_PORT_LINK_FLAP = a cable or port bouncing up and down.
- ADMIN_ACCESS = someone logged into the controller. Remote logins from unknown locations, unusual countries, ssh, or without MFA are suspicious, especially at night. Local logins and the home-assistant API key are routine.
- IPS_THREAT_DETECTED = intrusion prevention hit. Outbound malware / CnC / coinminer traffic from an internal client means that client may be compromised, even if blocked. Inbound scans that were blocked are background internet noise.
- ROGUE_AP_DETECTED = a foreign access point nearby; one broadcasting your own SSID (e.g. Home) with a strong signal is an evil-twin attack, a neighbour's router is noise.
House policy: page only for outages or likely compromises; notify for things that need a human today; put maintenance and firmware news in the daily digest; ignore routine churn. During a maintenance window, outages are expected.""",
    questions=[
        QuestionSpec("notify", "noul", [
            "Should the network admin get a notification for this event?",
            "Is this network event worth a push notification?",
            "Does someone need to know about this now?",
        ]),
        QuestionSpec("severity", "score", [
            "How severe is this network event?",
            "Rate the impact of this event on the home network.",
        ], ["noise", "informational", "minor", "degraded service", "outage or security incident"]),
        QuestionSpec("action", "choice", [
            "What should happen with this event?",
            "Pick the right handling for this event.",
        ], {"ignore": "drop it", "log": "keep it in the log only", "digest": "include in the daily summary",
            "notify": "push notification", "page": "urgent notification that breaks through do-not-disturb"}),
        QuestionSpec("category", "choice", [
            "What kind of event is this?",
            "Classify this network event.",
        ], {"client_churn": "routine client connect, disconnect or roam", "new_device": "a device not seen before",
            "internet": "WAN or ISP problem", "infrastructure": "a UniFi device or switch port problem",
            "security": "intrusion attempt, malware or suspicious access", "admin": "management access",
            "maintenance": "updates and housekeeping"}),
        QuestionSpec("is_security_relevant", "noul", [
            "Is this event security-relevant?",
            "Could this indicate a compromise or an intrusion attempt?",
        ]),
    ],
)
