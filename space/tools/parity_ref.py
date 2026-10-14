"""Python reference for the JS parity test (run by parity_test.mjs; usable on its own).

    .venv/bin/python space/tools/parity_ref.py --model artifacts/reef-tiny --per-pack 20 \
        --groups data,edge,long,many --out space/tools/.cache/ref.jsonl

For each case it records what the JS port must reproduce: the rendered state text, the request
after hoisting (state text, instructions, state token ids, number of state regions), every window
`Reef.decide()` runs in order (round, token ids, roles, levels, a hash of the int8 encoder input,
the raw int8 encoder output), the final unrounded per-option probabilities, and the public
response. The windows and probabilities are captured from inside `decide()` itself, so knockout
rounds, state chunking and hoisting are covered exactly as the runtime runs them.

Case groups:
  data   the first N of each data/v2-test/<pack>.jsonl file (states passed as dicts, like the server)
  edge   hand-written cases (passed as JSON text) that stress rendering/tokenising
  long   long states (1,500+ tokens) read in overlapping chunks, with and without gaps
  many   many-option questions (BANKING77-like 77, CLINC-like 151) answered as a knockout
  hoist  over-long instructions moved into the state (needs a model with hoist_instructions=true)
  v2     v2 encoder features: empty-state hoisting, opaque option keys (`option_N` -> description),
         even knockout splits, several question bands per window (meant for a v2 model)
  homelab  the first N per workflow of data/homelab-test.jsonl (states passed as dicts)
  banking  20 BANKING77-style 77-option intent cases (intent-name lists, `option_N` keys with
         descriptions, the message in the state or in the instructions with an empty state)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

import reef.runtime as reef_runtime
from reef import Reef
from reef.packing import hoist_groups, normalize_questions
from reef.text import render_state

REPO = Path(__file__).resolve().parents[2]

# (state JSON/plain text, questions JSON text). Questions text is parsed with json.loads (keeps order).
EDGE_CASES = [
    (
        '{"b": 1, "2": "two", "10": "ten", "a": {"1": true, "0": null}, "n": [1, 2.5, -0, -0.0, 1.0, 3]}',
        '{"7": {"type": "choice", "criteria": {"3": "three", "1": "one", "2": ""}}, "x": {"type": "noul"}}',
    ),
    (
        '{"floats": [0.0001234, 0.00001, 1234567.0, 1234500.0, 2.0001, 0.0625, 2.675, 1e20, 1.5e16, 999999.9996, 0.0005, -0.00049, 123456.789, 1e-320],'
        ' "big": 123456789012345678901234567890, "neg": -12, "e": 1E5, "nan": NaN, "inf": -Infinity}',
        '{"severity": {"type": "score", "criteria": {"10": "ten", "2": "two", "0": "zero"}}}',
    ),
    (
        '{"text": "Crème brûlée à la carte, ÀÉÎÕÜ ñ ß İstanbul ΣΑΣ 東京タワー 🙂 emoji\\\\u0000nul\\\\u200bzwsp\\\\ttab [CLS] [SEP][PAD]x [UNK]",'
        ' "ctl": "a\\\\u0007b\\\\u00adc\\\\ufffdd\\\\u0085e\\\\u3000f\\\\u2028g", "punct": "¿¡ «quoted» — dash … $5+3=8 ^_^ ~`|"}',
        '{"q": {"type": "noul", "instructions": "  Is it   spicy?  ", "criteria": {"TRUE": "hot", "False": 0}}}',
    ),
    (
        '{"messages": [{"role": "customer", "text": "  hi there  "}, {"author": "agent_bot", "content": "hello"},'
        ' {"role": "", "text": "anon"}, {"role": 5, "text": {"nested": [1, "x"]}}, {"role": "x", "text": "y", "extra": 1}],'
        ' "empty_obj": {}, "empty_list": [], "mixed": [1, [2, 3], {"k": "v"}], "deep": [[1, 2], [[]]], "s": "  padded  "}',
        '{"lvl9": {"type": "score", "criteria": ["a", "b", "c", "d", "e", "f", "g", "h", "i"]},'
        ' "lvl4": {"type": "score", "criteria": ["a", "b", "c", "d"]}, "pick": {"type": "choice", "criteria": ["red_one", "blue-two", 3]}}',
    ),
    ("plain text state, not JSON: disk sda1 is 97% full!", '{"urgent": {"type": "noul", "instructions": ""}}'),
    ('{"broken json": ', '{"ok": {"type": "noul"}}'),
    ('[{"a": 1}, {"b": [true, false, null]}]', '{"q_one": {"type": "choice", "criteria": {"yes_please": "", "no": "nope"}}}'),
    (
        # long state (two state regions since d182fb0) and a schema that needs two windows
        json.dumps({"log": " ".join(f"line{i} disk error on sda{i % 7} code {i * 37}" for i in range(120))}),
        json.dumps({f"q{i}": {"type": "choice", "instructions": "Which subsystem is failing here, in your judgement?",
                              "criteria": {f"opt_{j}": "a fairly long description of this option " * 2 for j in range(6)}}
                    for i in range(4)}),
    ),
]

BANKING77 = """activate_my_card age_limit apple_pay_or_google_pay atm_support automatic_top_up
balance_not_updated_after_bank_transfer balance_not_updated_after_cheque_or_cash_deposit beneficiary_not_allowed
cancel_transfer card_about_to_expire card_acceptance card_arrival card_delivery_estimate card_linking
card_not_working card_payment_fee_charged card_payment_not_recognised card_payment_wrong_exchange_rate
card_swallowed cash_withdrawal_charge cash_withdrawal_not_recognised change_pin compromised_card
contactless_not_working country_support declined_card_payment declined_cash_withdrawal declined_transfer
direct_debit_payment_not_recognised disposable_card_limits edit_personal_details exchange_charge exchange_rate
exchange_via_app extra_charge_on_statement failed_transfer fiat_currency_support get_disposable_virtual_card
get_physical_card getting_spare_card getting_virtual_card lost_or_stolen_card lost_or_stolen_phone
order_physical_card passcode_forgotten pending_card_payment pending_cash_withdrawal pending_top_up
pending_transfer pin_blocked receiving_money Refund_not_showing_up request_refund reverted_card_payment?
supported_cards_and_currencies terminate_account top_up_by_bank_transfer_charge top_up_by_card_charge
top_up_by_cash_or_cheque top_up_failed top_up_limits top_up_reverted topping_up_by_card
transaction_charged_twice transfer_fee_charged transfer_into_account transfer_not_received_by_recipient
transfer_timing unable_to_verify_identity verify_my_identity verify_source_of_funds verify_top_up
virtual_card_not_working visa_or_mastercard why_verify_identity wrong_amount_of_cash_received
wrong_exchange_rate_for_cash_withdrawal""".split()

CLINC151 = """transfer transactions balance freeze_account pay_bill bill_balance bill_due interest_rate routing
min_payment order_checks pin_change report_fraud account_blocked spending_history credit_score report_lost_card
credit_limit rewards_balance new_card application_status card_declined international_fees apr redeem_rewards
credit_limit_change damaged_card replacement_card_duration improve_credit_score expiration_date recipe
restaurant_reviews calories nutrition_info restaurant_suggestion ingredients_list ingredient_substitution
cook_time food_last meal_suggestion restaurant_reservation confirm_reservation how_busy cancel_reservation
accept_reservations shopping_list shopping_list_update next_song play_music update_playlist todo_list
todo_list_update calendar calendar_update what_song order order_status reminder reminder_update smart_home
traffic directions gas gas_type distance current_location mpg oil_change_when oil_change_how jump_start uber
schedule_maintenance last_maintenance tire_pressure tire_change book_flight book_hotel car_rental
travel_suggestion travel_alert travel_notification carry_on timezone vaccines translate flight_status
international_visa lost_luggage plug_type exchange_rate time alarm share_location find_phone weather text
spelling make_call timer date calculator measurement_conversion flip_coin roll_dice definition direct_deposit
pto_request taxes payday w2 pto_balance pto_request_status next_holiday insurance insurance_change
schedule_meeting pto_used meeting_schedule rollover_401k income greeting goodbye tell_joke where_are_you_from
how_old_are_you what_is_your_name who_made_you thank_you what_can_i_ask_you what_are_your_hobbies
do_you_have_pets are_you_a_bot meaning_of_life who_do_you_work_for fun_fact change_ai_name change_user_name
cancel user_name reset_settings whisper_mode repeat no yes maybe change_language change_accent change_volume
change_speed sync_device oos""".split()
assert len(BANKING77) == 77 == len(set(BANKING77)), len(BANKING77)
assert len(CLINC151) == 151 == len(set(CLINC151)), len(CLINC151)

WORDS = ("server disk alert customer order refund invoice payment card account delivery package router "
         "printer backup database latency error warning timeout network cluster node memory cpu ticket "
         "manager weekend holiday kitchen garden river mountain storm harbour lantern violin market").split()


def _sentences(rng: random.Random, n: int) -> list[str]:
    out = []
    for _ in range(n):
        k = rng.randint(6, 16)
        words = [rng.choice(WORDS) for _ in range(k)]
        if rng.random() < 0.3:
            words.insert(rng.randrange(k), str(rng.randint(1, 99999)))
        if rng.random() < 0.15:
            words.insert(rng.randrange(k), rng.choice(["café", "naïve", "Zürich", "東京", "🙂", "—", "e-mail", "O'Brien"]))
        s = " ".join(words)
        out.append(s[0].upper() + s[1:] + rng.choice([".", ".", "!", "?"]))
    return out


def _story(rng: random.Random, paras: int, per: int) -> list[str]:
    return [" ".join(_sentences(rng, per)) for _ in range(paras)]


def long_cases() -> list[tuple[str, dict, dict]]:
    rng = random.Random(7)
    chat = {"ticket": "T-4821", "channel": "chat",
            "messages": [{"role": "customer" if i % 2 == 0 else "agent", "text": " ".join(_sentences(rng, 3))}
                         for i in range(40)]}
    q1 = {
        "angry": {"type": "noul", "instructions": "Is the customer angry or frustrated?"},
        "topic": {"type": "choice", "instructions": "What is the conversation mainly about?",
                  "criteria": {"billing": "charges, refunds, invoices", "technical": "something is broken",
                               "delivery": "shipping and parcels", "account": "login, profile, closing"}},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["not urgent", "low", "medium", "high", "critical"]},
    }
    log = {"host": "db-03", "lines": [f"{i:05d} {rng.choice(['INFO', 'WARN', 'ERROR'])} " + _sentences(rng, 1)[0]
                                      for i in range(400)]}
    q2 = {
        "outage": {"type": "noul", "instructions": "Does the log show a service outage?"},
        "component": {"type": "choice", "criteria": ["disk", "memory", "network", "cpu", "database", "none_of_these"]},
    }
    text = "\n".join(_story(rng, 12, 6))
    q3 = {
        "region": {"type": "choice", "instructions": "Which department should handle this report?",
                   "criteria": [f"department_{c}" for c in "abcdefghijkl"]},
        "positive": {"type": "noul", "instructions": "Is the overall tone positive?"},
    }
    return [("long-chat", chat, q1), ("long-log-gaps", log, q2), ("long-plain", text, q3)]


def many_cases() -> list[tuple[str, dict, dict]]:
    rng = random.Random(11)
    msg = {"customer_message": "I topped up with my card yesterday but the money still has not arrived and the app says pending.",
           "account_age_days": 412}
    b77 = {"intent": {"type": "choice", "instructions": "Which banking intent does the customer message express?",
                      "criteria": BANKING77}}
    utter = {"utterance": "can you remind me to call the insurance company tomorrow at nine", "locale": "en-US"}
    c151 = {
        "intent": {"type": "choice", "instructions": "Classify the user's request into one intent.",
                   "criteria": {k: (f"the user wants {k.replace('_', ' ')}" if i % 3 == 0 else "")
                                for i, k in enumerate(CLINC151)}},
        "polite": {"type": "noul", "instructions": "Is the request phrased politely?"},
        "effort": {"type": "score", "criteria": ["trivial", "easy", "moderate", "hard", "very hard"]},
    }
    long_state = {"thread": [{"role": "customer", "text": " ".join(_sentences(rng, 4))} for _ in range(25)]}
    combo = {
        "bank": {"type": "choice", "criteria": BANKING77},
        "clinc": {"type": "choice", "instructions": "Closest assistant intent?", "criteria": CLINC151},
        "lvl9": {"type": "score", "instructions": "Severity on a nine point scale.",
                 "criteria": [f"level {i}" for i in range(9)]},
        "spam": {"type": "noul"},
    }
    sizes = {
        "eight": {"type": "choice", "criteria": [f"option_{i}" for i in range(8)]},
        "nine": {"type": "choice", "criteria": [f"option_{i}" for i in range(9)]},
        "sixteen": {"type": "choice", "criteria": {f"label_{i}": f"description number {i} of this label" for i in range(16)}},
        "score12": {"type": "score", "criteria": [f"step {i}" for i in range(12)]},
    }
    return [("many-banking77", msg, b77), ("many-clinc151", utter, c151), ("many-combo-long", long_state, combo),
            ("many-sizes", {"note": "boundary sizes around eight options per part"}, sizes)]


def hoist_cases() -> list[tuple[str, object, dict]]:
    rng = random.Random(5)
    story = _story(rng, 8, 7)
    question = "Question: based on the story above, did the main character finally reach the harbour?"
    single = {"answer": {"type": "noul", "instructions": "\n\n".join(story + [question])}}
    lines = {"answer": {"type": "choice", "instructions": "\n".join(_sentences(rng, 40) + ["Which item matters most?"]),
                        "criteria": ["the_router", "the_printer", "the_database", "nothing"]}}
    blob = {"answer": {"type": "score", "instructions": " ".join(_sentences(rng, 30)),
                       "criteria": ["none", "a little", "some", "a lot", "all"]}}
    prefix = _story(rng, 4, 6)
    tools = ["web_search", "calculator", "calendar", "send_email", "weather_lookup"]
    shared = {f"use_{t}": {"type": "noul", "instructions": "\n\n".join(prefix + [f"Candidate tool: {t}"])} for t in tools}
    shared["best_tool"] = {"type": "choice", "instructions": "\n\n".join(prefix + ["Which tool should be called first?"]),
                           "criteria": tools}
    distinct = {f"q{i}": {"type": "noul", "instructions": " ".join(_sentences(rng, 12))} for i in range(3)}
    seps = ["\r\n\r\n", "\n \u00a0\t\n", "\n\u3000\n", "\n\u0085\n", "\n\ufeff\n", "\n\u2028\n", "\n\x1c\n"]
    odd = "".join(p + seps[i % len(seps)] for i, p in enumerate(_story(rng, 9, 4))) + "Is anything on fire?"
    odd_q = {"fire": {"type": "noul", "instructions": odd}}
    many = {"intent": {"type": "choice", "instructions": "\n\n".join(_story(rng, 6, 6) + ["Which intent fits best?"]),
                       "criteria": CLINC151[:20]}}
    two_fit = {"answer": {"type": "noul", "instructions": "\n\n".join(_story(rng, 5, 6) + [
        "Context: the reader only cares about the last day of the trip.", "Did it rain?"])}}

    # ToolRet-like: a shared task paragraph, then one long candidate document per question (one
    # paragraph, or several), and a short question mixed in -> per-question groups.
    task = ("You are given a user query and the documentation of one candidate tool. Decide whether the "
            "tool can be used to answer the query. Query: book a table for four in Lisbon tonight.")
    def doc(name: str, paras: int) -> str:
        return "\n\n".join([f"Tool {name}."] + _story(rng, paras, 5)) if paras > 1 else f"Tool {name}. " + " ".join(_sentences(rng, 18))
    toolret = {
        "restaurant_booker": {"type": "noul", "instructions": f"{task}\n\n{doc('restaurant_booker', 1)}"},
        "flight_search": {"type": "noul", "instructions": f"{task}\n\n{doc('flight_search', 3)}"},
        "short_one": {"type": "noul", "instructions": f"{task}\n\nTool clock: tells the time."},
        "maps": {"type": "noul", "instructions": f"{task}\n\n{doc('maps', 1)}"},
        "weather": {"type": "score", "instructions": f"{task}\n\n{doc('weather', 2)}",
                    "criteria": ["useless", "marginal", "partly useful", "useful", "exactly right"]},
        "calendar": {"type": "noul", "instructions": f"{task}\n\n{doc('calendar', 1)}"},
    }
    toolret_mixed = {
        "short_choice": {"type": "choice", "instructions": "Which tool fits best?", "criteria": ["booker", "maps", "none"]},
        "doc_a": {"type": "noul", "instructions": f"Is this tool relevant?\n\n{doc('alpha', 1)}"},
        "doc_b": {"type": "noul", "instructions": f"Does this tool need an API key?\n\n{doc('beta', 2)}"},
        "short_noul": {"type": "noul", "instructions": "Is the query about food?"},
    }
    task_doc = {"relevant": {"type": "noul", "instructions": f"{task}\n\n{doc('restaurant_booker', 1)}"}}
    two_task_doc = {"relevant": {"type": "choice", "instructions": "\n\n".join([
        "Read the document below.", "Then decide how relevant it is to the query about Lisbon.", doc("guide", 1)]),
        "criteria": {"high": "directly answers", "low": "barely related", "none": "unrelated"}}}
    long_task_doc = {"relevant": {"type": "noul", "instructions": " ".join(_sentences(rng, 14)) + "\n\n" + doc("x", 1)}}
    return [
        ("hoist-single-empty-state", "", single),
        ("hoist-single-none-state", "{}", single),
        ("hoist-single-with-state", '{"user": "ana", "plan": "pro"}', single),
        ("hoist-lines", '{"ctx": "lines only"}', lines),
        ("hoist-blob-stub", "", blob),
        ("hoist-shared-prefix", '{"user_request": "what is the weather in Zürich tomorrow?"}', shared),
        ("hoist-distinct-long", "", distinct),
        ("hoist-odd-separators", "none", odd_q),
        ("hoist-many-options", "", many),
        ("hoist-two-fit", "", two_fit),
        ("hoist-toolret", '{"session": "s-19", "locale": "pt-PT"}', toolret),
        ("hoist-toolret-empty-state", "", toolret),
        ("hoist-toolret-mixed", '{"query": "book a table for four in Lisbon tonight"}', toolret_mixed),
        ("hoist-task-doc", "", task_doc),
        ("hoist-task-doc-with-state", '{"query": "table for four"}', task_doc),
        ("hoist-two-task-paras-doc", "", two_task_doc),
        ("hoist-long-task-doc-stub", "", long_task_doc),
    ]


def v2_cases() -> list[tuple[str, object, dict]]:
    """Cases for v2 encoders (bands, hoist_empty_state, drop_opaque_keys, even knockout splits)."""
    rng = random.Random(13)
    banking = {"intent": {"type": "choice", "instructions": (
        "Which banking intent does this customer message express? Message: I topped up with my card "
        "yesterday but the money still has not arrived and the app says pending."),
        "criteria": {f"option_{i + 1}": f"the customer asks about {k.rstrip('?').replace('_', ' ').lower()}"
                     for i, k in enumerate(BANKING77)}}}
    task = ("You are given a support ticket. Read it carefully and answer the question below.\n\n"
            "Ticket: " + " ".join(_sentences(rng, 6)))
    shared = {
        "angry": {"type": "noul", "instructions": f"{task}\n\nIs the customer angry?"},
        "topic": {"type": "choice", "instructions": f"{task}\n\nWhat is the ticket about?",
                  "criteria": {"option_1": "billing and refunds", "option_2": "a technical fault",
                               "option_3": "delivery", "option_4": "the account itself"}},
        "urgency": {"type": "score", "instructions": f"{task}\n\nHow urgent is it?",
                    "criteria": {"0": "not urgent", "1": "low", "2": "medium", "3": "high"}},
    }
    distinct = {
        "spam": {"type": "noul", "instructions": "Is this message spam? Message: " + " ".join(_sentences(rng, 2))},
        "lang": {"type": "choice", "instructions": "Which language is this? Text: Bonjour, je voudrais un café.",
                 "criteria": {"A": "English", "B": "French", "C": "German", "D": "Spanish"}},
        "stars": {"type": "score", "instructions": "How many stars does this review deserve? " + " ".join(_sentences(rng, 3)),
                  "criteria": ["one", "two", "three", "four", "five"]},
    }
    blank_instr = {"a": {"type": "noul"}, "b": {"type": "choice", "criteria": {"x1": "yes", "x2": "no"}}}
    mixed = {"pick": {"type": "choice", "instructions": "Which option fits the note best?", "criteria": {
        "A": "the first option", "b": "the second option", "3": "the third option", "Option-2": "the fourth option",
        "label 7": "the fifth option", "refund": "money back", "item_5": "", "choice10": "the tenth option",
        "\u212a": "kelvin sign key", "\u017f": "long s key", "\u0130": "dotted capital i key", "\u0661\u0662": "arabic digits key",
        "option_\u0663": "arabic three", "\u00b2": "superscript two", "opt 9": "a very long description " * 12,
        "class_\U0001d7ce": "math digit zero", "AB": "two letters", "candidate-": "dangling dash"}}}
    nine = {"nine": {"type": "choice", "instructions": "Which department should handle this?",
                     "criteria": {f"option_{i}": f"department number {i} which handles {w}" for i, w in
                                  enumerate(["billing", "repairs", "shipping", "accounts", "fraud", "sales",
                                             "returns", "legal", "other"])}},
            "nine_list": {"type": "choice", "criteria": [f"team_{c}" for c in "abcdefghi"]}}
    seventeen = {"route": {"type": "choice", "instructions": "Route the ticket to one queue.",
                           "criteria": {f"queue_{i}": " ".join(_sentences(rng, 1)) for i in range(17)}}}
    long_state = {"thread": [{"role": "customer" if i % 2 == 0 else "agent", "text": " ".join(_sentences(rng, 3))}
                             for i in range(30)]}
    five = {
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "topic": {"type": "choice", "criteria": {"1": "billing", "2": "technical", "3": "delivery", "4": "account"}},
        "urgency": {"type": "score", "criteria": [f"level {i}" for i in range(5)]},
        "resolved": {"type": "noul", "instructions": "Was the issue resolved by the end?"},
        "next": {"type": "choice", "instructions": "What should happen next?",
                 "criteria": {"opt1": "close the ticket", "opt2": "escalate", "opt3": "refund", "opt4": "wait"}},
    }
    return [
        ("v2-banking77-empty", "", banking),
        ("v2-banking77-with-state", '{"channel": "app"}', banking),
        ("v2-shared-prefix-empty", "", shared),
        ("v2-shared-prefix-null", "null", shared),
        ("v2-distinct-empty", "", distinct),
        ("v2-distinct-colon-none", ": none", distinct),
        ("v2-distinct-none-text", "none", distinct),
        ("v2-distinct-with-state", '{"user": "ana"}', distinct),
        ("v2-blank-instructions-empty", "{}", blank_instr),
        ("v2-mixed-keys", '{"note": "customer wants the money back"}', mixed),
        ("v2-mixed-keys-empty", "", mixed),
        ("v2-nine-even-split", '{"ticket": "my parcel never arrived and I was charged twice"}', nine),
        ("v2-seventeen", '{"ticket": "printer on floor 3 is jammed again"}', seventeen),
        ("v2-long-state-five", long_state, five),
        ("v2-long-state-banking", long_state, banking),
    ]


BANKING_MESSAGES = [
    "I topped up with my card yesterday but the money still has not arrived and the app says pending.",
    "My new card still hasn't shown up, it has been ten days since I ordered it.",
    "Why was I charged a fee when I paid with my card abroad?",
    "Someone stole my wallet this morning, please block my card right now.",
    "The ATM gave me 50 euros less than I asked for.",
    "How do I change the PIN on my card?",
    "I was charged twice for the same coffee.",
    "Can I get a virtual card for online shopping?",
    "My transfer to my landlord hasn't arrived yet, it was sent on Monday.",
    "What exchange rate do you use when I take out cash in dollars?",
    "I want to close my account, how do I do that?",
    "Apple Pay stopped working on my phone after the update.",
    "Is there an age limit to open an account? My son is 15.",
    "The app won't let me verify my identity, the photo keeps failing.",
    "There is a payment on my statement I don't recognise from a shop in Lyon.",
    "How long does a bank transfer from the UK usually take?",
    "I forgot my passcode and I'm locked out of the app.",
    "My card got declined at the supermarket even though I have money.",
    "Can I top up my account with cash at a shop?",
    "I requested a refund from a merchant two weeks ago and it's still not showing.",
]


def banking_cases() -> list[tuple[str, object, dict]]:
    names = BANKING77
    desc = {k: k.rstrip("?").replace("_", " ").lower() for k in names}
    out = []
    for i, msg in enumerate(BANKING_MESSAGES):
        kind = i % 4
        if kind == 0:  # intent names as a list, message in the state
            out.append((f"banking-{i:02d}-list", {"customer_message": msg, "channel": "app"},
                        {"intent": {"type": "choice", "instructions": "Which banking intent does the message express?",
                                    "criteria": names}}))
        elif kind == 1:  # opaque keys with descriptions, message in the state
            out.append((f"banking-{i:02d}-opaque", {"message": msg},
                        {"intent": {"type": "choice", "instructions": "Classify the customer's request.",
                                    "criteria": {f"option_{j + 1}": f"the customer asks about {desc[k]}" for j, k in enumerate(names)}}}))
        elif kind == 2:  # named keys with descriptions, message in the instructions, empty state
            out.append((f"banking-{i:02d}-empty-named", "",
                        {"intent": {"type": "choice", "instructions": f"Customer message: {msg}\n\nWhich intent fits best?",
                                    "criteria": {k: desc[k] for k in names}}}))
        else:  # opaque keys, empty state, a second question sharing the message paragraph
            out.append((f"banking-{i:02d}-empty-two", "",
                        {"intent": {"type": "choice", "instructions": f"Customer message: {msg}\n\nWhich intent fits best?",
                                    "criteria": {f"option_{j}": desc[k] for j, k in enumerate(names)}},
                         "urgent": {"type": "noul", "instructions": f"Customer message: {msg}\n\nDoes this need an urgent reply?"}}))
    return out


def run_case(reef: Reef, case_id: str, state, questions: dict, state_text: str, questions_text: str) -> dict:
    rendered = render_state(state)
    qs = normalize_questions(questions)
    groups = [(list(range(len(qs))), qs, rendered)]
    if reef.hoist:
        groups = hoist_groups(qs, rendered, lambda t: len(reef.packer.encode(t)), reef.instruction_cap,
                              empty=reef.hoist_empty)
    planned = []
    for idx, gqs, gstate in groups:
        state_ids = reef.packer.encode(gstate)
        regions, gaps = reef.packer.state_regions(state_ids, reef.max_state_windows)
        planned.append({"idx": idx, "state_text": gstate, "instructions": [q.instructions for q in gqs],
                        "state_ids": state_ids, "state_regions": len(regions), "gaps": gaps})

    # Capture every window decide() runs (with its group and round) and the final probabilities.
    windows, probs, where = [], {}, {"group": -1, "round": -1}
    orig_run, orig_score, orig_group = reef.run_window, reef._score_parts, reef._answer
    orig_answer = reef_runtime.make_answer

    def run_window(ids, roles, levels):
        x = reef.embed(ids, roles, levels).reshape(1, *reef.grid, -1)
        xq = np.clip(np.round(x / reef._in_scale + reef._in_zp), -128, 127).astype(reef._in_dtype)
        reef._interp.set_tensor(reef._in_idx, xq)
        reef._interp.invoke()
        y = reef._interp.get_tensor(reef._out_idx).copy()
        logits = orig_run(ids, roles, levels)
        assert np.array_equal(logits, ((y.astype(np.float32) - reef._out_zp) * reef._out_scale).reshape(-1))
        windows.append({
            **where, "ids": ids.tolist(), "roles": roles.tolist(), "levels": levels.tolist(),
            "input_sha256": hashlib.sha256(xq.tobytes()).hexdigest(), "output": y.reshape(-1).astype(int).tolist(),
        })
        return logits

    def score_parts(regs, parts):
        where["round"] += 1
        return orig_score(regs, parts)

    def answer_group(gqs, gstate):
        where["group"] += 1
        where["round"] = -1
        return orig_group(gqs, gstate)

    def make_answer(q, p):
        assert q.id not in probs
        probs[q.id] = [float(v) for v in p]
        return orig_answer(q, p)

    reef.run_window, reef._score_parts, reef._answer = run_window, score_parts, answer_group
    reef_runtime.make_answer = make_answer
    try:
        decided = reef.decide(state, questions)
    finally:
        del reef.run_window, reef._score_parts, reef._answer
        reef_runtime.make_answer = orig_answer
    assert where["group"] == len(groups) - 1
    return {
        "id": case_id, "state_text": state_text, "questions_text": questions_text, "rendered": rendered,
        "groups": planned, "windows": windows,
        "options": {q.id: [o.id for o in q.options] for q in qs},
        "probs": probs, "answers": decided["answers"], "usage": decided["usage"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "artifacts" / "reef-tiny"))
    ap.add_argument("--data", default=str(REPO / "data" / "v2-test"))
    ap.add_argument("--per-pack", type=int, default=20, help="cases per pack (0 = all)")
    ap.add_argument("--groups", default="data,edge,long,many", help="comma-separated case groups")
    ap.add_argument("--homelab", default=str(REPO / "data" / "homelab-test.jsonl"))
    ap.add_argument("--homelab-per", type=int, default=10, help="homelab cases per workflow")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    groups = args.groups.split(",")

    reef = Reef(args.model, device="cpu", strict=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:

        def emit(rec: dict, group: str, pack: str) -> None:
            nonlocal n
            rec["group"], rec["pack"] = group, pack
            f.write(json.dumps(rec) + "\n")
            n += 1

        if "data" in groups:
            for path in sorted(Path(args.data).glob("*.jsonl")):
                lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
                if args.per_pack:
                    lines = lines[: args.per_pack]
                for ln in lines:
                    c = json.loads(ln)
                    rec = run_case(reef, c["id"], c["state"], c["questions"], json.dumps(c["state"], ensure_ascii=False),
                                   json.dumps(c["questions"], ensure_ascii=False))
                    emit(rec, "data", c.get("workflow", path.stem))
        if "homelab" in groups:
            seen: dict[str, int] = {}
            for ln in Path(args.homelab).read_text().splitlines():
                if not ln.strip():
                    continue
                c = json.loads(ln)
                w = c.get("workflow", "homelab")
                if seen.get(w, 0) >= args.homelab_per:
                    continue
                seen[w] = seen.get(w, 0) + 1
                rec = run_case(reef, c["id"], c["state"], c["questions"], json.dumps(c["state"], ensure_ascii=False),
                               json.dumps(c["questions"], ensure_ascii=False))
                emit(rec, "homelab", w)
        if "edge" in groups:
            for i, (state_text, questions_text) in enumerate(EDGE_CASES):
                emit(run_case(reef, f"edge-{i}", state_text, json.loads(questions_text), state_text, questions_text),
                     "edge", "edge")
        for group, make in (("long", long_cases), ("many", many_cases), ("hoist", hoist_cases), ("v2", v2_cases),
                            ("banking", banking_cases)):
            if group not in groups:
                continue
            for case_id, state, questions in make():
                # States and questions go to both runtimes as JSON text (a str state stays plain text).
                state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
                questions_text = json.dumps(questions, ensure_ascii=False)
                emit(run_case(reef, case_id, state_text, json.loads(questions_text), state_text, questions_text),
                     group, group)
    print(f"wrote {n} reference cases to {out} (model {reef.name}, device {reef.device}, hoist {reef.hoist}, "
          f"hoist empty {reef.hoist_empty}, bands {reef.packer.bands}, instruction cap {reef.instruction_cap})")


if __name__ == "__main__":
    main()
