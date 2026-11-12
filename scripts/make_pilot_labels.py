"""Pilot labels for 40 of the 100 test-batch reviews, made in Claude Code (not the API), saved in the
same format as data/labels_final.jsonl. They exist only to build and test the Phase 3 code before the
paid run. They are NOT used for any reported result.

Each entry: review_id -> {"neg": [(category, evidence)], "pos": [...]}
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import batch_labels as bl

P = {
    153: {"neg": [("other", "The design is too kitsch to me")],
          "pos": [("room", "Bathroom Jacuzzi bath tub and shower cabin plenty of towels and toiletries"),
                  ("staff", "Great staff"), ("location", "very good location in a quiet area"),
                  ("quiet", "quiet area"), ("other", "Locked securely at night 24 7 reception")]},
    255: {"neg": [("facilities", "High priced car parking"), ("value", "High priced car parking"),
                  ("breakfast", "Limited dinner menu the meals where not that great")]},
    981: {"neg": [("cleanliness", "There was a funny smell in our room"), ("facilities", "Too warm in the lobby"),
                  ("wifi", "Wifi was slow")],
          "pos": [("location", "Excellent location"), ("staff", "Wonderful porters Nice staff"),
                  ("bed_comfort", "Comfy bed"), ("facilities", "Loved the coffee and drinks bar in the lobby")]},
    1131: {"neg": [("facilities", "There was no fridge in the room")],
           "pos": [("room", "The room was spacious"), ("bed_comfort", "three comfortable huge beds"),
                   ("cleanliness", "The bathroom was very clean"),
                   ("location", "Location very close to the direct train to the city centre")]},
    1903: {"neg": [("cleanliness", "Room wasn t so clean"), ("maintenance", "A lot of things was broken and old"),
                   ("bed_comfort", "Absolutely not comfortable bed and sheets"),
                   ("value", "they tried to scam me for 45 euros")],
           "pos": [("room", "Room wasn t small")]},
    2388: {"neg": [("staff", "Unpolite staff no customer orientation at all")]},
    3261: {"neg": [("checkin", "when we were checking out they said it was a mistake"),
                   ("staff", "they recognized it and then only let us know at the very end")],
           "pos": [("breakfast", "Gorgeous breakfast"), ("location", "excellent location")]},
    3503: {"neg": [("wifi", "Wifi not consistent"), ("facilities", "Pool and area is small and was crowded at times")],
           "pos": [("room", "The balcony attached to the room"), ("design", "The decor was beautiful"),
                   ("staff", "Staff were helpful")]},
    4081: {"neg": [("location", "A bit far from the city center")],
           "pos": [("room", "Spacious room Separated shower and bath"),
                   ("other", "Can redeem traveler check without commission fee")]},
    4552: {"neg": [("bed_comfort", "The poor bedding"), ("maintenance", "plumbing to be entirely replaced"),
                   ("noise", "noise from the street")],
           "pos": [("location", "The location"), ("room", "the size of the room")]},
    4661: {"neg": [("room_size", "Tiny room"), ("cleanliness", "Very dusty Smelly shower curtain"),
                   ("value", "Very expensive for a very poor room")],
           "pos": [("staff", "Helpful staff")]},
    4768: {"neg": [("wifi", "Wifi was really bad in the room"), ("noise", "there was really loud noise all day"),
                   ("value", "the price was still very high despite this renovation")]},
    4855: {"neg": [("room_size", "The room is small"), ("facilities", "There is no drinking water"),
                   ("noise", "There is no soundproofing")],
           "pos": [("staff", "The staff treated was very nice")]},
    4937: {"neg": [("checkin", "Check in took unacceptably long long queue"),
                   ("maintenance", "Bathroom very tired with temperamental shower"),
                   ("other", "Fire alarm went off at 3 15am"),
                   ("staff", "no apology or any form of compensation from hotel")],
           "pos": [("location", "Good location"), ("facilities", "Nice public spaces good for business meetings")]},
    5102: {"neg": [("checkin", "Waiting times to check In and out"),
                   ("room_size", "the size of the room for the price was very poor"),
                   ("value", "the size of the room for the price was very poor")],
           "pos": [("design", "The Hotel is beautiful"), ("location", "very well located")]},
    5380: {"neg": [("other", "There was no light by the bed"), ("staff", "They took my own private washing cloth"),
                   ("temperature", "There was a cold draught from the window"),
                   ("staff", "Why didn t they ask me what it was all about"), ("wifi", "Internet for GBP 9 99")],
           "pos": [("location", "Great location")]},
    5715: {"neg": [("room_size", "Room very small")], "pos": [("cleanliness", "Clean"), ("staff", "good staff")]},
    5903: {"neg": [("value", "Very expensive for what it was")], "pos": [("location", "The location")]},
    6285: {"neg": [("maintenance", "Dodgy lift")], "pos": [("location", "location"), ("staff", "friendly staff")]},
    6393: {"neg": [], "pos": [("other", "The whole experience of staying at this hotel was great")]},
    7707: {"neg": [("facilities", "there is nowhere to sit for paying guests staying overnight")],
           "pos": [("cleanliness", "Room was very clean"), ("bed_comfort", "nice comfortable bed"),
                   ("room", "great blackout blinds"), ("location", "Location very good"),
                   ("other", "This was a great Hotel")]},
    7820: {"neg": [("room_size", "Room was tiny even though I paid for a king size room"),
                   ("value", "Room was tiny even though I paid for a king size room"),
                   ("maintenance", "Bathroom was in a poor condition"),
                   ("facilities", "Bar closed at 10 30pm so couldn t get a drink"),
                   ("staff", "Receptionist was very blunt")]},
    8235: {"neg": [("checkin", "we re told our room was not ready and to come back in an hour"),
                   ("maintenance", "the light up number outside our room was not connected wires on show"),
                   ("cleanliness", "The bathroom was filthy"), ("other", "The view was directly into flats"),
                   ("value", "even though our room was not cheap"),
                   ("noise", "we were woken by workmen drilling outside the door")],
           "pos": [("staff", "The staff that worked on the entrance and took your bags were fantastic")]},
    9090: {"neg": [("facilities", "no access to normal tv channels only paid for available")],
           "pos": [("cleanliness", "cleanliness"), ("breakfast", "bar restaurant"), ("facilities", "bar restaurant")]},
    10368: {"neg": [("bed_comfort", "Hard pillows"), ("noise", "Noise from corridor near lifts"),
                    ("breakfast", "Breakfast been kept warm too long Sausage and bacon over cooked")],
            "pos": [("location", "Nice location")]},
    11589: {"neg": [("breakfast", "It would have been nice if the price had included breakfast"),
                    ("value", "as it wasn t cheap")],
            "pos": [("room", "Rooms are spacious"), ("bed_comfort", "comfy beds"),
                    ("quiet", "nice thick walls so you can t hear anything from other rooms"),
                    ("location", "Brilliant location for sightseeing and easy underground access"),
                    ("cleanliness", "Everything was really clean")]},
    11887: {"neg": [], "pos": [("breakfast", "Breakfast was top class")]},
    12337: {"neg": [("staff", "Staff not helpful")]},
    13127: {"neg": [("room_size", "The rooms are tiny"), ("breakfast", "the food is inedible"),
                    ("maintenance", "the equipment airco furniture etc has seen much better days"),
                    ("value", "for a 4 star establishment you would maybe have expected something better")],
            "pos": [("location", "The hotel was relatively well situated"),
                    ("staff", "the staff was very helpful and friendly")]},
    13771: {"neg": [("wifi", "No free wifi in room"), ("cleanliness", "Long dark hair in bath tub"),
                    ("other", "Poor shampoo etc and only one bar if soap"),
                    ("maintenance", "Awful colours in bedroom greens and beige"),
                    ("facilities", "Fridge with do not move items or you ll pay regardless of not drinking them"),
                    ("value", "or you ll pay regardless of not drinking them"),
                    ("facilities", "No complimentary water")],
            "pos": [("staff", "Welcoming receptionist"), ("room", "Large bedroom"), ("quiet", "Quiet"),
                    ("location", "Easy inside access from hotel into St Pancras train station"),
                    ("room", "Very soft bathrobe")]},
    14331: {"neg": [("bed_comfort", "found the bed uncomfortable"),
                    ("cleanliness", "the room not as clean as the last time")],
            "pos": [("staff", "Staff very helpful and friendly")]},
    14514: {"neg": [("temperature", "room had to be changed as AirCon was not working"),
                    ("value", "price i discovered later was higher than others around")],
            "pos": [("location", "Location")]},
    15387: {"neg": [("location", "A bit of a distance from main attractions in Milan")],
            "pos": [("design", "Modern hotel"), ("cleanliness", "very clean"),
                    ("staff", "very friendly and helpful staff"), ("location", "Walking distance from bars in Navigli")]},
    16612: {"neg": [("temperature", "No air conditioning"), ("maintenance", "Safety deposit box was not working"),
                    ("staff", "Rude staff")],
            "pos": [("bed_comfort", "Comfy bed"), ("room", "Lovely shower"), ("staff", "Receptionist friendly")]},
    17005: {"neg": [("noise", "got a street facing which was somewhat noisy"),
                    ("checkin", "requesting a quite room facing the rear and got a street facing")],
            "pos": [("design", "Good boutique hotel"), ("value", "reasonably priced for Paris"),
                    ("location", "within walking quick metro distance of some tourist spots"),
                    ("breakfast", "Breakfast on the pricey side but was good"),
                    ("room", "Bedroom was very comfortable as was the shower room"),
                    ("wifi", "Wifi had good speed in the bedroom")]},
    17128: {"neg": [("checkin", "we were accommodated in the wrong room for our first night")],
            "pos": [("location", "Good location near St Germain area"), ("room", "Nice big room with good storage"),
                    ("bed_comfort", "comfy bed")]},
    17796: {"neg": [("staff", "Staff were unhelpful at times and impolite")],
            "pos": [("cleanliness", "The rooms were nice and clean"),
                    ("location", "Location is okay very close to a metro station")]},
    17943: {"neg": [], "pos": [("location", "Location was great"), ("staff", "staff very friendly and helpful")]},
    18394: {"neg": [("facilities", "no place to sit with my friends for evening drink"),
                    ("staff", "They made me beg to use a dinner table")],
            "pos": [("checkin", "Staff let me check in early"), ("staff", "Front desk fabulous")]},
    19431: {"neg": [("cleanliness", "The room was dirty"), ("breakfast", "The breakfast was plain"),
                    ("facilities", "The parking advertised as free was not available at night"),
                    ("value", "for a small fee not cool")],
            "pos": [("location", "It is close to Mariahilfe strasse")]},
}


def main():
    clean = bl.load_clean_text().set_index("review_id")
    with open(bl.PILOT_PATH, "w", encoding="utf-8") as f:
        for rid, boxes in P.items():
            for box in bl.BOXES:
                if not clean.at[rid, f"{box}_clean"].strip():
                    assert box not in boxes, f"{rid} {box} is empty but has labels"
                    continue
                labels = [{"category": c, "evidence": e} for c, e in boxes.get(box, [])]
                rec = bl.result_record(bl.custom_id(box, rid), "succeeded", labels, stop_reason="end_turn")
                rec["source"] = "claude-code-pilot"
                f.write(json.dumps(rec) + "\n")
    print(f"Wrote {bl.PILOT_PATH}")


if __name__ == "__main__":
    main()
