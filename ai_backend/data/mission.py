#!/usr/bin/env python3
"""
create_mission_files.py

Creates one text file per mission in an output folder ("missions_output").
Uses a supplied 'mission.txt' (if present) to include Apollo project text as an Apollo file.
Generates a curated set of well-known missions and programmatically creates additional entries
to reach 100 total. Each file is a single paragraph containing:
- Mission name
- Country
- Date
- Outcome
- Type
- Description (<= ~200 words)

Run: python3 create_mission_files.py
"""

import os
import datetime
import random
import textwrap
import re

OUTPUT_DIR = "missions_output"
INPUT_APOLLO_FILE = "mission.txt"
TOTAL_MISSIONS = 100
MAX_DESC_WORDS = 200

random.seed(42)

# A starter set of well-known missions (real). Add more real entries here if desired.
KNOWN_MISSIONS = [
    {"name": "Apollo_11", "country": "USA", "date": "1969-07-16", "outcome": "Passed", "type": "Manned lunar landing"},
    {"name": "Apollo_13", "country": "USA", "date": "1970-04-11", "outcome": "Partial Success", "type": "Manned lunar (aborted)"},
    {"name": "Sputnik_1", "country": "USSR", "date": "1957-10-04", "outcome": "Passed", "type": "Unmanned orbital (first artificial satellite)"},
    {"name": "Vostok_1", "country": "USSR", "date": "1961-04-12", "outcome": "Passed", "type": "Manned orbital (first human in space)"},
    {"name": "Luna_2", "country": "USSR", "date": "1959-09-12", "outcome": "Passed", "type": "Unmanned lunar impactor"},
    {"name": "Luna_9", "country": "USSR", "date": "1966-02-03", "outcome": "Passed", "type": "Unmanned lunar soft lander"},
    {"name": "Chandrayaan-1", "country": "India", "date": "2008-10-22", "outcome": "Passed", "type": "Unmanned lunar orbiter"},
    {"name": "Chandrayaan-2_Orbiter", "country": "India", "date": "2019-07-22", "outcome": "Passed (orbiter); lander failed", "type": "Unmanned lunar orbiter + lander"},
    {"name": "Chandrayaan-3", "country": "India", "date": "2023-07-14", "outcome": "Passed", "type": "Unmanned lunar lander + rover"},
    {"name": "Aditya-L1", "country": "India", "date": "2023-09-02", "outcome": "Passed", "type": "Unmanned solar observatory (L1)"},
    {"name": "Kaguya_SELENE", "country": "Japan", "date": "2007-09-14", "outcome": "Passed", "type": "Unmanned lunar orbiter"},
    {"name": "Hayabusa", "country": "Japan", "date": "2003-05-09", "outcome": "Passed", "type": "Unmanned asteroid sample return"},
    {"name": "Hayabusa2", "country": "Japan", "date": "2014-12-03", "outcome": "Passed", "type": "Unmanned asteroid sample return"},
    {"name": "SMART-1", "country": "ESA", "date": "2003-09-27", "outcome": "Passed", "type": "Unmanned lunar orbiter / tech demonstrator"},
    {"name": "Chang_e_3", "country": "China", "date": "2013-12-01", "outcome": "Passed", "type": "Unmanned lunar lander + rover"},
    {"name": "Chang_e_4", "country": "China", "date": "2018-12-07", "outcome": "Passed", "type": "Unmanned far-side lunar lander + rover"},
    {"name": "Chang_e_5", "country": "China", "date": "2020-11-23", "outcome": "Passed", "type": "Unmanned lunar sample return"},
    {"name": "Beresheet", "country": "Israel", "date": "2019-02-21", "outcome": "Failed", "type": "Unmanned lunar lander (crash)"},
    {"name": "LADEE", "country": "USA (NASA)", "date": "2013-09-06", "outcome": "Passed", "type": "Unmanned lunar exosphere mission"},
    {"name": "Voyager_1", "country": "USA (NASA)", "date": "1977-09-05", "outcome": "Passed", "type": "Unmanned planetary / interstellar probe"},
    {"name": "Pioneer_10", "country": "USA (NASA)", "date": "1972-03-02", "outcome": "Passed", "type": "Unmanned Jupiter flyby / interstellar probe"},
    {"name": "Mars_Pathfinder", "country": "USA (NASA)", "date": "1996-12-04", "outcome": "Passed", "type": "Unmanned Mars lander + rover"},
    {"name": "Curiosity_MSL", "country": "USA (NASA)", "date": "2011-11-26", "outcome": "Passed", "type": "Unmanned Mars rover"},
    {"name": "Perseverance", "country": "USA (NASA)", "date": "2020-07-30", "outcome": "Passed", "type": "Unmanned Mars rover + sample caching"},
    {"name": "Rosetta", "country": "ESA", "date": "2004-03-02", "outcome": "Passed", "type": "Unmanned comet rendezvous and lander"},
    {"name": "Venera_7", "country": "USSR", "date": "1970-12-15", "outcome": "Passed", "type": "Unmanned Venus lander"},
    {"name": "Mariner_2", "country": "USA (NASA)", "date": "1962-08-27", "outcome": "Passed", "type": "Unmanned Venus flyby"},
    {"name": "Zond_5", "country": "USSR", "date": "1968-09-15", "outcome": "Passed", "type": "Unmanned circumlunar (biological payload)"},
    {"name": "Mir", "country": "USSR/Russia", "date": "1986-02-20", "outcome": "Passed", "type": "Crewed space station"},
    {"name": "ISS_Zarya_core", "country": "International", "date": "1998-11-20", "outcome": "Passed", "type": "Crewed space station (international)"},
]

# Utility: safe filename
def safe_filename(name):
    # replace spaces and forbidden chars with underscore; limit length
    name = re.sub(r"[^\w\-_.()]", "_", name)
    return name[:200] + ".txt"

# Load mission.txt content if present
def load_apollo_text(path):
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None
    return None

# Generate additional synthetic but plausible missions to reach the TOTAL_MISSIONS count
def generate_additional_missions(start_index, count):
    countries = ["USA", "USSR/Russia", "India", "China", "Japan", "ESA", "Israel", "UK", "France", "Germany", "Canada", "Australia", "UAE", "South Korea", "Brazil", "Argentina"]
    types = [
        "Unmanned lunar orbiter", "Unmanned lunar lander", "Manned orbital",
        "Unmanned Mars orbiter", "Unmanned Mars lander", "Planetary flyby",
        "Unmanned asteroid mission", "Solar observatory (L1/L2)", "Technology demonstrator",
        "Crewed lunar mission", "Lunar sample return", "Deep space probe", "Space telescope"
    ]
    outcomes = ["Passed", "Failed", "Partial Success"]
    extra = []
    year = 1970
    for i in range(count):
        idx = start_index + i
        country = random.choice(countries)
        mtype = random.choice(types)
        # create a readable mission name
        short_country = country.split()[0]
        name = f"{short_country}_Mission_{idx}"
        # Year progression roughly, cap at 2025
        year += random.choice([0, 0, 1, 1, 2])
        year = min(year, 2025)
        date = f"{year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        outcome = random.choices(outcomes, weights=[0.78, 0.12, 0.10])[0]
        extra.append({"name": name, "country": country, "date": date, "outcome": outcome, "type": mtype})
    return extra

# Description generator (<= ~200 words)
def generate_description(m):
    templates = [
        "{name} was a {type} launched by {country} on {date}.",
        "Launched on {date}, {name} from {country} served as a {type}.",
        "{country}'s {name}, launched {date}, performed a {type} mission."
    ]
    mid = [
        "The mission's objectives included scientific observation, technology demonstration, and data collection to support future exploration.",
        "Primary goals were to map surface composition, test landing and navigation systems, characterize the local environment, and return telemetry to ground stations.",
        "Engineers used this flight to validate hardware, improve communications, and gather operational experience for follow-up missions."
    ]
    outcome_text = {
        "Passed": "It achieved its objectives and provided valuable data that contributed to scientific knowledge and subsequent missions.",
        "Failed": "The mission encountered critical failures and did not meet its primary objectives, but lessons learned informed later programs.",
        "Partial Success": "Some objectives were met while others were not; the partial success still yielded scientific returns and engineering lessons."
    }
    extras = [
        "Onboard instruments typically included imagers, spectrometers, magnetometers, and sometimes landers or rovers for in-situ study.",
        "The mission added to the global understanding of planetary formation, surface processes, and potential resources.",
        "International collaborations were often leveraged for instrument contributions and data analysis."
    ]
    intro = random.choice(templates).format(**m)
    body = random.choice(mid)
    outcome = outcome_text.get(m.get("outcome", "Passed"), outcome_text["Passed"])
    extra = random.choice(extras)
    paragraph = " ".join([intro, body, outcome, extra])
    words = paragraph.split()
    if len(words) > MAX_DESC_WORDS:
        paragraph = " ".join(words[:MAX_DESC_WORDS]).rstrip() + "."
    paragraph = " ".join(paragraph.split())
    return paragraph

# Build single-paragraph file content
def build_paragraph(m):
    meta_parts = [
        f"Mission Name: {m.get('name','Unknown')}",
        f"Country: {m.get('country','Unknown')}",
        f"Date: {m.get('date','Unknown')}",
        f"Outcome: {m.get('outcome','Unknown')}",
        f"Type: {m.get('type','Unknown')}"
    ]
    meta = ". ".join(meta_parts) + "."
    desc = m.get("description") or generate_description(m)
    paragraph = f"{meta} Description: {desc}"
    # ensure single paragraph (no newlines)
    paragraph_one_line = " ".join(paragraph.split())
    # wrap for readability when writing file
    wrapped = textwrap.fill(paragraph_one_line, width=80)
    return wrapped

def write_mission_file(folder, mission):
    fname = safe_filename(mission["name"])
    path = os.path.join(folder, fname)
    content = build_paragraph(mission)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content + "\n")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    apollo_text = load_apollo_text(INPUT_APOLLO_FILE)

    missions = KNOWN_MISSIONS.copy()
    if len(missions) < TOTAL_MISSIONS:
        additional = generate_additional_missions(start_index=len(missions)+1, count=(TOTAL_MISSIONS - len(missions)))
        missions.extend(additional)

    # Ensure descriptions exist
    for m in missions:
        if "description" not in m or not m["description"]:
            m["description"] = generate_description(m)

    # If mission.txt exists, create a special Apollo_Project file
    if apollo_text:
        apollo_paragraph = " ".join(line.strip() for line in apollo_text.splitlines() if line.strip())
        apollo_content = textwrap.fill(apollo_paragraph, width=80)
        with open(os.path.join(OUTPUT_DIR, safe_filename("Apollo_Project")), "w", encoding="utf-8") as f:
            f.write(apollo_content + "\n")

    # Write one file per mission
    for m in missions:
        write_mission_file(OUTPUT_DIR, m)

    print(f"Created {len(missions) + (1 if apollo_text else 0)} files in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()