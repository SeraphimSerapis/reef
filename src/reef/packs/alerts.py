"""alert-triage: Alertmanager, Uptime Kuma and syslog events from a home lab."""

from __future__ import annotations

import random

from reef.packs.base import Pack, QuestionSpec, clock, human_duration, maybe, pick

HOSTS = {
    "nas01": "NAS (TrueNAS, family photos and backups)",
    "pve1": "Proxmox node running most VMs",
    "pve2": "secondary Proxmox node",
    "pihole": "Pi-hole DNS for the whole house",
    "opnsense": "router / firewall",
    "homeassistant": "Home Assistant VM",
    "jellyfin": "media server",
    "backup01": "offsite backup target",
    "k3s-w1": "k3s worker",
    "rpi-zigbee": "Raspberry Pi running Zigbee2MQTT",
    "unifi": "UniFi controller",
    "frigate": "Frigate NVR box",
}

SERVICES = ["nextcloud", "jellyfin", "immich", "vaultwarden", "paperless", "grafana", "gitea", "adguard",
            "home-assistant", "plex", "sonarr", "radarr", "traefik", "nginx-proxy-manager", "mosquitto", "frigate"]


def _alertmanager(rng: random.Random, host: str) -> dict:
    kind = pick(rng, {
        "disk": 4, "cpu": 2, "mem": 2, "down": 3, "smart": 2, "zfs": 2, "temp": 1.5, "ups": 1.5, "cert": 1.5,
        "backup": 2, "crashloop": 1.5, "watchdog": 0.7, "clock": 0.7, "net": 1.5,
    })
    labels = {"instance": f"{host}:9100", "job": "node"}
    ann: dict = {}
    if kind == "disk":
        mount = rng.choice(["/", "/var", "/mnt/tank", "/volume1", "/var/lib/docker", "/boot"])
        pct = rng.choice([80, 85, 88, 90, 93, 95, 97, 99])
        name = "NodeFilesystemAlmostOutOfSpace" if pct >= 93 else "NodeFilesystemSpaceFillingUp"
        hours = rng.choice([None, 4, 24, 72])
        ann = {"summary": f"Filesystem {mount} is {pct}% full",
               "description": f"{mount} on {host} has {100 - pct}% space left"
               + (f" and is predicted to fill up within {hours}h." if hours else ".")}
        sev = "critical" if pct >= 95 else "warning"
    elif kind == "cpu":
        name, sev = "HostHighCpuLoad", rng.choice(["warning", "info"])
        ann = {"summary": f"CPU load is {rng.randint(85, 100)}%",
               "description": rng.choice(["Load average above number of cores for 15m.",
                                          "Sustained high CPU, top process: ffmpeg (transcode).",
                                          "Sustained high CPU, top process: kswapd0.",
                                          "Sustained high CPU, top process: xmrig.",
                                          "Sustained high CPU during nightly scrub window."])}
    elif kind == "mem":
        name, sev = rng.choice([("HostOutOfMemory", "warning"), ("HostOomKillDetected", "warning")])
        ann = {"summary": f"Memory available {rng.randint(1, 9)}%",
               "description": rng.choice(["OOM killer invoked for java (pid 2211).", "Swap usage at 100%.",
                                          "Memory pressure high, ZFS ARC not shrinking.", "Container immich_ml killed (OOM)."])}
    elif kind == "down":
        name, sev = rng.choice([("TargetDown", "warning"), ("InstanceDown", "critical")])
        labels["job"] = rng.choice(["node", "cadvisor", "blackbox", "unpoller", "snmp"])
        ann = {"summary": f"{labels['job']} target on {host} is unreachable",
               "description": f"Scrape failed for {rng.choice([2, 5, 10, 30, 120])} minutes."}
    elif kind == "smart":
        name, sev = "SmartDeviceFailing", rng.choice(["warning", "critical"])
        attr = rng.choice(["Reallocated_Sector_Ct increased 0 -> 8", "Current_Pending_Sector = 24",
                           "UDMA_CRC_Error_Count increased by 3", "SMART overall-health self-assessment: FAILED",
                           "Power_On_Hours = 61000"])
        ann = {"summary": f"Disk /dev/sd{rng.choice('abcdef')} SMART warning", "description": attr}
    elif kind == "zfs":
        name, sev = rng.choice([("ZfsPoolDegraded", "critical"), ("ZfsScrubErrors", "warning"),
                                ("ZfsPoolResilvering", "info")])
        ann = {"summary": f"Pool tank is {name.removeprefix('ZfsPool').lower() or 'reporting errors'}",
               "description": rng.choice(["1 device FAULTED in mirror-0.", "Scrub repaired 128K with 0 errors.",
                                          "Resilver 43% done, 3h left.", "2 checksum errors on ata-WDC_WD80EFAX."])}
    elif kind == "temp":
        name, sev = "HostTemperatureHigh", rng.choice(["warning", "critical"])
        ann = {"summary": f"{rng.choice(['CPU', 'NVMe', 'HDD', 'chassis'])} temperature {rng.randint(62, 95)}°C",
               "description": rng.choice(["Fan 2 reports 0 RPM.", "Ambient is high (summer).", "Sustained over 15m."])}
    elif kind == "ups":
        name, sev = rng.choice([("UpsOnBattery", "critical"), ("UpsBatteryLow", "critical"),
                                ("UpsSelfTestFailed", "warning"), ("UpsOnline", "info")])
        ann = {"summary": name, "description": f"Battery {rng.randint(8, 100)}%, runtime {rng.randint(2, 45)} min."}
        labels["job"] = "nut"
    elif kind == "cert":
        days = rng.choice([1, 3, 7, 14, 21, 29])
        name, sev = "SSLCertExpiringSoon", "critical" if days <= 3 else "warning"
        ann = {"summary": f"Certificate for {rng.choice(SERVICES)}.home.example.com expires in {days} days",
               "description": rng.choice(["Renewal via ACME DNS-01 has been failing.", "Auto-renew scheduled.",
                                          "Issued by Let's Encrypt."])}
        labels["job"] = "blackbox"
    elif kind == "backup":
        name, sev = rng.choice([("BackupJobFailed", "warning"), ("BackupTooOld", "warning"),
                                ("BackupJobSucceeded", "info")])
        ann = {"summary": f"{rng.choice(['restic', 'borg', 'PBS', 'Kopia', 'rclone'])} job "
                          f"{rng.choice(['photos', 'vms', 'documents', 'db'])} {name.removeprefix('BackupJob').lower()}",
               "description": rng.choice(["Last successful run 9 days ago.", "Repository locked by another process.",
                                          "Target unreachable: no route to host.", "Completed in 42m, 12 GB added."])}
    elif kind == "crashloop":
        name, sev = "KubePodCrashLooping", "warning"
        svc = rng.choice(SERVICES)
        labels.update({"namespace": "apps", "pod": f"{svc}-{rng.randint(1000, 9999)}"})
        ann = {"summary": f"Pod {labels['pod']} restarting", "description":
               f"Restarted {rng.randint(3, 60)} times in the last hour. Last exit code {rng.choice([1, 137, 139, 2])}."}
    elif kind == "watchdog":
        name, sev = "Watchdog", "none"
        ann = {"summary": "Alerting pipeline heartbeat", "description": "This alert is always firing."}
    elif kind == "clock":
        name, sev = "NodeClockSkewDetected", "warning"
        ann = {"summary": f"Clock skew {rng.choice([0.07, 0.3, 2.5, 40])}s", "description": "NTP not synchronised."}
    else:  # net
        name, sev = rng.choice([("HostNetworkReceiveErrors", "warning"), ("WanLatencyHigh", "warning"),
                                ("WanDown", "critical"), ("HostNetworkInterfaceFlapping", "warning")])
        ann = {"summary": name, "description": rng.choice(["Packet loss 12% to 1.1.1.1.", "eth0 link up/down 6 times.",
                                                           "ISP gateway not responding.", "rx_errors 3400 in 5m."])}
    labels.update({"alertname": name, "severity": sev})
    return {
        "source": "alertmanager",
        "status": pick(rng, {"firing": 0.85, "resolved": 0.15}),
        "labels": labels,
        "annotations": ann,
        "firing_for": human_duration(rng.choice([45, 300, 900, 3600, 7200, 21600, 86400, 259200])),
    }


def _uptime_kuma(rng: random.Random, host: str) -> dict:
    svc = rng.choice(SERVICES)
    status = pick(rng, {"down": 0.75, "up": 0.25})
    msg = rng.choice([
        "timeout of 48000ms exceeded", "Request failed with status code 502", "Request failed with status code 503",
        "getaddrinfo ENOTFOUND", "connect ECONNREFUSED 10.0.10.21:443", "certificate has expired",
        "Request failed with status code 401", "keyword 'OK' not found", "ping: 100% packet loss",
    ]) if status == "down" else rng.choice(["200 - OK", "Up after downtime"])
    return {
        "source": "uptime-kuma",
        "monitor": f"{svc} ({rng.choice(['https', 'http', 'ping', 'keyword', 'docker'])})",
        "host": host,
        "status": status,
        "message": msg,
        "down_for": human_duration(rng.choice([30, 60, 120, 600, 1800, 7200, 43200])) if status == "down" else None,
        "flaps_last_24h": rng.choice([0, 0, 0, 1, 2, 5, 14]),
    }


def _external_ip(rng: random.Random) -> str:
    """A random 'internet' address from the RFC 5737 documentation ranges, never a real host. Draws
    four numbers like a fully random IPv4 would, so seeded datasets stay reproducible."""
    a, _b, _c, d = rng.randint(1, 223), rng.randint(0, 255), rng.randint(0, 255), rng.randint(1, 254)
    return f"{DOC_NETS[a % 3]}{d}"


DOC_NETS = ("192.0.2.", "198.51.100.", "203.0.113.")


def _syslog(rng: random.Random, host: str) -> dict:
    line = rng.choice([
        f"sshd[{rng.randint(900, 9999)}]: Failed password for root from {_external_ip(rng)} port {rng.randint(30000, 65000)} ssh2",
        f"sshd[{rng.randint(900, 9999)}]: Accepted publickey for admin from 10.0.10.{rng.randint(2, 254)} port 51022 ssh2",
        f"sshd[{rng.randint(900, 9999)}]: Accepted password for root from {_external_ip(rng)} port 40112 ssh2",
        f"kernel: [{rng.randint(1000, 999999)}.123] blk_update_request: I/O error, dev sd{rng.choice('abcd')}, sector {rng.randint(10**6, 10**9)}",
        "kernel: Out of memory: Killed process 4121 (qemu-system-x86) total-vm:17GB",
        f"smartd[812]: Device: /dev/sd{rng.choice('abcd')} [SAT], {rng.choice([1, 8, 40])} Currently unreadable (pending) sectors",
        f"systemd[1]: {rng.choice(SERVICES)}.service: Main process exited, code=exited, status=1/FAILURE",
        "systemd[1]: Started Daily apt download activities.",
        f"dockerd[1022]: container {rng.choice(SERVICES)} exited with code {rng.choice([0, 1, 137, 143])}",
        "zed[2210]: eid=118 class=statechange pool='tank' vdev=ata-ST8000VN004 vdev_state=FAULTED",
        "CRON[3311]: (root) CMD (/usr/local/bin/backup.sh)",
        f"kernel: e1000e 0000:00:1f.6 eno1: NIC Link is {rng.choice(['Down', 'Up 1000 Mbps Full Duplex'])}",
        "kernel: mce: [Hardware Error]: Machine check events logged",
        "pihole-FTL[633]: Rate-limiting 10.0.20.44 for at least 60 seconds",
        "sudo: admin : TTY=pts/0 ; PWD=/home/admin ; USER=root ; COMMAND=/usr/bin/apt upgrade",
        "fail2ban.actions[901]: NOTICE [sshd] Ban 198.51.100.4",
        "nut-monitor[702]: UPS ups@localhost on battery",
        "certbot[5520]: Certificate not yet due for renewal",
    ])
    return {"source": "syslog", "host": host, "line": line,
            "repeats_last_hour": rng.choice([1, 1, 1, 2, 5, 30, 400])}


def generate(rng: random.Random) -> dict:
    host = rng.choice(list(HOSTS))
    src = pick(rng, {"am": 0.55, "kuma": 0.2, "syslog": 0.25})
    event = _alertmanager(rng, host) if src == "am" else _uptime_kuma(rng, host) if src == "kuma" else _syslog(rng, host)
    when, hour = clock(rng)
    context = {
        "local_time": when,
        "host_role": HOSTS[host],
        "maintenance_window": maybe(rng, 0.1, True, False),
        "someone_home": maybe(rng, 0.7, True, False),
    }
    if rng.random() < 0.2:
        context["recent_changes"] = rng.choice(["kernel upgrade 2h ago", "moved rack yesterday", "replaced disk sdb today",
                                                "power cut 30 min ago", "none"])
    return {**event, "context": context}


PACK = Pack(
    name="alert-triage",
    description="Triage Alertmanager, Uptime Kuma and syslog events from a home lab.",
    generate=generate,
    tags=["alertmanager", "uptime-kuma", "syslog", "prometheus"],
    teacher_notes="""\
Context: a home lab run by one or two people in their spare time. Nobody is paid to be on call.
Glossary: Alertmanager alerts have a status (firing / resolved), labels (alertname, severity, instance) and how long they have been firing. Uptime Kuma reports monitor up/down and for how long. Syslog lines come straight from hosts (kernel, smartd, zfs, systemd, fail2ban, nut, certbot ...).
House policy:
- Resolved alerts, heartbeats, recoveries and routine log lines are noise.
- Data-loss risks come first: failing or FAILED disks (SMART, reallocated or pending sectors), degraded RAID/ZFS pools, filesystems nearly full on storage that matters, backups that stopped running. These are actionable, and a pool that is degraded or a disk that has failed is worth waking up for.
- UPS on battery is urgent only if the outage is long or the battery is low; a short blip that recovered is noise.
- A single service down for a minute or two is usually a blip; down for a long time, or a core service (DNS, reverse proxy, Home Assistant, the NAS) is actionable.
- Certificates expiring in weeks are low priority; expiring within a few days, or already expired, is high.
- fail2ban bans and blocked scans are routine security noise; successful logins from unknown places are not.
- Only wake someone at night for data at risk, the whole network or home automation down, or safety (overheating, power).""",
    questions=[
        QuestionSpec("actionable", "noul", [
            "Does this need a human to do something?",
            "Is this alert actionable?",
            "Should someone act on this event?",
        ]),
        QuestionSpec("severity", "score", [
            "How urgent is this?",
            "Rate the severity of this event.",
            "How quickly does this need attention?",
        ], ["ignore, noise", "low, look this week", "medium, look today", "high, look within the hour",
            "critical, drop everything"]),
        QuestionSpec("category", "choice", [
            "What kind of problem is this?",
            "Which area does this event belong to?",
        ], {"storage": "disks, filesystems, pools, SMART", "network": "connectivity, DNS, WAN, interfaces",
            "compute": "CPU, memory, processes", "service": "an application or container is unhealthy",
            "security": "logins, intrusions, bans, suspicious activity",
            "power_hardware": "UPS, temperature, fans, hardware errors",
            "certificates": "TLS certificates and renewal", "backup": "backup jobs and their age",
            "informational": "heartbeats, recoveries, routine messages"}),
        QuestionSpec("wake_me", "noul", [
            "Is this worth waking someone up at night?",
            "Should this page the on-call person even at 3 am?",
        ]),
        QuestionSpec("likely_noise", "noul", [
            "Is this most likely a transient blip or a false alarm?",
            "Will this probably resolve itself without intervention?",
        ]),
        QuestionSpec("data_at_risk", "noul", [
            "Is there a risk of losing data?",
            "Could this lead to data loss if ignored?",
        ]),
    ],
)
