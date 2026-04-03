import os
import sys
import pickle
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from sgp4.api import Satrec, jday

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_tle(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    lines = [l.strip() for l in r.text.strip().splitlines() if l.strip()]
    sats = []
    for i in range(0, len(lines)-2, 3):
        name = lines[i]
        l1, l2 = lines[i+1], lines[i+2]
        if l1.startswith("1") and l2.startswith("2"):
            sats.append({"name": name, "line1": l1, "line2": l2})
    return sats

def extract_features(line2_iss, line2_debris):
    def parse(l2):
        return [
            float(l2[8:16]),
            float(l2[17:25]),
            float("0." + l2[26:33]),
            float(l2[34:42]),
            float(l2[43:51]),
            float(l2[52:63]),
        ]
    p = parse(line2_iss)
    d = parse(line2_debris)
    return [
        p[0], p[1], p[2], p[3], p[4], p[5],
        d[0], d[1], d[2], d[3], d[4], d[5],
        abs(p[0]-d[0]), abs(p[1]-d[1]), abs(p[2]-d[2]),
        abs(p[3]-d[3]), abs(p[4]-d[4]), abs(p[5]-d[5]),
    ]

def get_positions(line1, line2, times):
    sat = Satrec.twoline2rv(line1, line2)
    positions = []
    for dt in times:
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        e, r, v = sat.sgp4(jd, fr)
        positions.append(np.array(r)*1000 if e == 0 else None)
    return positions

def min_distance(pos1, pos2):
    dists = [np.linalg.norm(p1-p2) for p1, p2 in zip(pos1, pos2)
             if p1 is not None and p2 is not None]
    return min(dists) if dists else float('inf')

# Step 1: Load Random Forest
print("="*60)
print("STEP 1: Loading Random Forest model...")
print("="*60)
with open("random_forest_model.pkl", "rb") as f:
    rf = pickle.load(f)

# Step 2: Fetch live data
print("\nSTEP 2: Fetching live TLE data from CelesTrak...")
iss_list = fetch_tle("https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE")
iss = iss_list[0]
print(f"Protected satellite: {iss['name']}")

debris_list = fetch_tle("https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=TLE")
print(f"Debris objects fetched: {len(debris_list)}")

# Step 3: Screen with Random Forest
print("\nSTEP 3: Screening with Random Forest...")
now = datetime.now(timezone.utc)
times = [now + timedelta(seconds=30*i) for i in range(240)]
iss_positions = get_positions(iss["line1"], iss["line2"], times)

high_risk = []
for debris in debris_list:
    features = extract_features(iss["line2"], debris["line2"])
    prob = rf.predict_proba([features])[0][1]
    if rf.predict([features])[0] == 1:
        debris_positions = get_positions(debris["line1"], debris["line2"], times)
        dist = min_distance(iss_positions, debris_positions)
        high_risk.append({
            "name": debris["name"],
            "line1": debris["line1"],
            "line2": debris["line2"],
            "risk_prob": prob,
            "min_dist_km": dist/1000
        })

high_risk.sort(key=lambda x: x["min_dist_km"])
print(f"High risk objects found: {len(high_risk)}")
print(f"Closest threat: {high_risk[0]['min_dist_km']:.1f} km away")
print(f"Top 5 threats:")
for i, h in enumerate(high_risk[:5]):
    print(f"  #{i+1} Min distance: {h['min_dist_km']:.1f} km | Risk: {h['risk_prob']:.3f}")

# Step 4: Generate collision scenario
print("\nSTEP 4: Generating collision environment...")
env_path = "data/environments/rf_pipeline.env"
model_path = "training/agents_tables/CE/rf_pipeline_model.csv"

os.system(
    f"python generation/generate_collision.py "
    f"-save_path {env_path} "
    f"-n_d 5 "
    f"-start 6601 "
    f"-end 6601.1 "
    f"-before 0.1"
)
print(f"Environment saved to {env_path}")

# Step 5: Train RL avoidance maneuver
print("\nSTEP 5: Training RL avoidance maneuver (Cross Entropy)...")
os.system(
    f"python training/CE/CE_train_for_collision.py "
    f"-env {env_path} "
    f"-save_path {model_path} "
    f"-r false -n_m 1 -print true"
)

# Step 6: Run simulation with visualization
print("\nSTEP 6: Running simulation with visualization...")
print("Close the plot window when done.")
os.system(
    f"python examples/collision.py "
    f"-env {env_path} "
    f"-model {model_path} "
    f"-p true -print true -n_v 5000"
)

print("\nPipeline complete!")