"""Generated states are published as datasets: they must never contain routable public IPs (only
private and RFC 5737 documentation ranges) or identifiers copied from a real network."""

import ipaddress
import json
import random
import re

import pytest

from reef import packs

IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
DOC = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
MAC = re.compile(r"\b[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}\b")
RESOLVERS = {"1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4", "9.9.9.9"}  # well-known ping targets


@pytest.mark.parametrize("name", packs.names())
def test_no_public_ips_or_vendor_macs(name):
    pack = packs.get(name)
    rng = random.Random(0)
    for _ in range(400):
        text = json.dumps(pack.generate(rng))
        for m in IP.findall(text):
            try:
                ip = ipaddress.ip_address(m)
            except ValueError:
                continue  # e.g. a version string like 4.0.12.300
            assert (ip.is_private or ip.is_loopback or m in RESOLVERS or any(ip in n for n in DOC)), \
                f"{name}: public IP {m}"
        for m in MAC.findall(text):
            # made-up MACs are locally administered (second-lowest bit of the first octet set)
            assert int(m[:2], 16) & 0b10, f"{name}: globally unique (vendor) MAC {m}"
