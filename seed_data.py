#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency. Install with: pip install numpy pandas") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed synthetic race-team data into SQLite.")
    p.add_argument("--db-path", default="race_team.db")
    p.add_argument("--seasons", type=int, default=2)
    p.add_argument("--races-per-season", type=int, default=24)
    p.add_argument("--teams", type=int, default=4)
    p.add_argument("--drivers-per-team", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-year", type=int, default=2024)
    return p.parse_args()


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(
        """
        DROP TABLE IF EXISTS pit_stops;
        DROP TABLE IF EXISTS lap_times;
        DROP TABLE IF EXISTS race_results;
        DROP TABLE IF EXISTS weather;
        DROP TABLE IF EXISTS races;
        DROP TABLE IF EXISTS cars;
        DROP TABLE IF EXISTS drivers;
        DROP TABLE IF EXISTS teams;

        CREATE TABLE teams (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            base_country TEXT NOT NULL,
            budget_musd REAL NOT NULL,
            car_performance REAL NOT NULL
        );

        CREATE TABLE drivers (
            id INTEGER PRIMARY KEY,
            team_id INTEGER NOT NULL,
            full_name TEXT NOT NULL,
            nationality TEXT NOT NULL,
            experience_years INTEGER NOT NULL,
            skill_rating REAL NOT NULL,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        );

        CREATE TABLE cars (
            id INTEGER PRIMARY KEY,
            team_id INTEGER NOT NULL,
            season_year INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            reliability REAL NOT NULL,
            top_speed_kph REAL NOT NULL,
            UNIQUE(team_id, season_year),
            FOREIGN KEY (team_id) REFERENCES teams(id)
        );

        CREATE TABLE races (
            id INTEGER PRIMARY KEY,
            season_year INTEGER NOT NULL,
            round_number INTEGER NOT NULL,
            race_name TEXT NOT NULL,
            circuit TEXT NOT NULL,
            race_date TEXT NOT NULL,
            laps INTEGER NOT NULL,
            UNIQUE(season_year, round_number)
        );

        CREATE TABLE weather (
            id INTEGER PRIMARY KEY,
            race_id INTEGER NOT NULL UNIQUE,
            condition TEXT NOT NULL,
            air_temp_c REAL NOT NULL,
            track_temp_c REAL NOT NULL,
            rain_probability REAL NOT NULL,
            FOREIGN KEY (race_id) REFERENCES races(id)
        );

        CREATE TABLE race_results (
            id INTEGER PRIMARY KEY,
            race_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            car_id INTEGER NOT NULL,
            start_position INTEGER NOT NULL,
            finish_position INTEGER NOT NULL,
            status TEXT NOT NULL,
            points REAL NOT NULL,
            total_time_sec REAL,
            fastest_lap_sec REAL,
            UNIQUE(race_id, driver_id),
            FOREIGN KEY (race_id) REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id),
            FOREIGN KEY (car_id) REFERENCES cars(id)
        );

        CREATE TABLE lap_times (
            id INTEGER PRIMARY KEY,
            race_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            lap_number INTEGER NOT NULL,
            lap_time_sec REAL NOT NULL,
            UNIQUE(race_id, driver_id, lap_number),
            FOREIGN KEY (race_id) REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id)
        );

        CREATE TABLE pit_stops (
            id INTEGER PRIMARY KEY,
            race_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            stop_number INTEGER NOT NULL,
            lap_number INTEGER NOT NULL,
            duration_sec REAL NOT NULL,
            UNIQUE(race_id, driver_id, stop_number),
            FOREIGN KEY (race_id) REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id)
        );

        CREATE INDEX idx_results_race ON race_results(race_id);
        CREATE INDEX idx_results_driver ON race_results(driver_id);
        CREATE INDEX idx_laps_race_driver ON lap_times(race_id, driver_id);
        CREATE INDEX idx_pits_race_driver ON pit_stops(race_id, driver_id);
        """
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    db = Path(args.db_path)

    season_years = [args.start_year + i for i in range(args.seasons)]

    team_names = ["Apex Racing", "Velocity Racing", "Titan Racing", "Quantum Racing", "Falcon Racing", "Vortex Racing"]
    countries = ["UK", "Italy", "Germany", "USA", "Japan", "France"]
    teams = []
    for i in range(args.teams):
        teams.append({
            "id": i + 1,
            "name": team_names[i % len(team_names)],
            "base_country": countries[i % len(countries)],
            "budget_musd": round(float(rng.uniform(180, 450)), 1),
            "car_performance": round(float(rng.uniform(0.82, 1.12)), 3),
        })
    teams_df = pd.DataFrame(teams)

    first_names = ["Liam", "Noah", "Emma", "Olivia", "Ava", "Elijah", "Lucas", "Mia", "Ethan", "Sofia", "Logan", "Leo"]
    last_names = ["Parker", "Rossi", "Schmidt", "Tanaka", "Dubois", "Khan", "Anders", "Silva", "Novak", "Costa", "Murphy", "Ivanov"]
    nats = ["UK", "Italy", "Germany", "Brazil", "Japan", "Spain", "USA", "France"]
    drivers = []
    d_id = 1
    for team_id in teams_df["id"].tolist():
        for _ in range(args.drivers_per_team):
            drivers.append({
                "id": d_id,
                "team_id": int(team_id),
                "full_name": f"{random.choice(first_names)} {random.choice(last_names)}",
                "nationality": random.choice(nats),
                "experience_years": int(rng.integers(0, 13)),
                "skill_rating": round(float(rng.uniform(0.82, 1.15)), 3),
            })
            d_id += 1
    drivers_df = pd.DataFrame(drivers)

    cars = []
    c_id = 1
    for season in season_years:
        for _, t in teams_df.iterrows():
            cars.append({
                "id": c_id,
                "team_id": int(t["id"]),
                "season_year": int(season),
                "model_name": f"{t['name'].split()[0]}-{str(season)[-2:]}",
                "reliability": round(float(rng.uniform(0.86, 0.995)), 3),
                "top_speed_kph": round(float(rng.uniform(325, 354)), 1),
            })
            c_id += 1
    cars_df = pd.DataFrame(cars)

    circuits = ["Silverstone", "Monza", "Suzuka", "Spa", "Austin", "Barcelona", "Montreal", "Imola", "Interlagos", "Zandvoort"]
    races = []
    r_id = 1
    for season in season_years:
        start = date(season, 3, 1)
        for rnd in range(1, args.races_per_season + 1):
            race_date = start + timedelta(days=14 * (rnd - 1))
            circuit = circuits[(rnd - 1) % len(circuits)]
            races.append({
                "id": r_id,
                "season_year": season,
                "round_number": rnd,
                "race_name": f"{circuit} Grand Prix",
                "circuit": circuit,
                "race_date": race_date.isoformat(),
                "laps": int(rng.integers(52, 71)),
            })
            r_id += 1
    races_df = pd.DataFrame(races)

    conditions = ["Clear", "Cloudy", "Windy", "Light Rain", "Heavy Rain"]
    probs = np.array([0.34, 0.26, 0.16, 0.18, 0.06])
    weather = []
    for _, race in races_df.iterrows():
        cond = str(rng.choice(conditions, p=probs))
        if cond == "Clear":
            rain_prob = float(rng.uniform(0.0, 0.1))
        elif cond == "Cloudy":
            rain_prob = float(rng.uniform(0.05, 0.3))
        elif cond == "Windy":
            rain_prob = float(rng.uniform(0.0, 0.25))
        elif cond == "Light Rain":
            rain_prob = float(rng.uniform(0.5, 0.8))
        else:
            rain_prob = float(rng.uniform(0.75, 1.0))
        air = float(rng.uniform(14, 33))
        weather.append({
            "id": int(race["id"]),
            "race_id": int(race["id"]),
            "condition": cond,
            "air_temp_c": round(air, 1),
            "track_temp_c": round(air + float(rng.uniform(4, 18)), 1),
            "rain_probability": round(rain_prob, 3),
        })
    weather_df = pd.DataFrame(weather)

    points = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    team_perf = teams_df.set_index("id")["car_performance"].to_dict()
    car_lookup = cars_df.set_index(["team_id", "season_year"])["id"].to_dict()
    weather_map = weather_df.set_index("race_id").to_dict("index")
    driver_rows = drivers_df.to_dict("records")

    results, laps, pits = [], [], []
    res_id = lap_id = pit_id = 1

    for _, race in races_df.iterrows():
        rid = int(race["id"])
        season = int(race["season_year"])
        total_laps = int(race["laps"])
        rain_factor = 1.0 + float(weather_map[rid]["rain_probability"]) * 0.09

        entries = []
        for d in driver_rows:
            tid = int(d["team_id"])
            expected = 92.5 - (float(d["skill_rating"]) * 3.6 + float(team_perf[tid]) * 2.8) + rng.normal(0, 0.35)
            entries.append({
                "driver_id": int(d["id"]),
                "team_id": tid,
                "car_id": int(car_lookup[(tid, season)]),
                "expected": expected,
            })

        start_grid = sorted(entries, key=lambda x: x["expected"] + rng.normal(0, 0.5))
        for pos, e in enumerate(start_grid, start=1):
            e["start_position"] = pos

        finish = sorted(entries, key=lambda x: x["expected"] * rain_factor + rng.normal(0, 0.9))

        for finish_pos, e in enumerate(finish, start=1):
            base_lap = e["expected"] * rain_factor
            series = np.clip(rng.normal(base_lap, 0.95, total_laps), 75.0, 130.0)
            fastest = float(np.min(series))
            total_time = float(np.sum(series))

            car_rel = float(cars_df.loc[cars_df["id"] == e["car_id"], "reliability"].iloc[0])
            dnf = bool(rng.random() < max(0.01, 0.16 - car_rel * 0.12))
            status = "DNF" if dnf else "Finished"
            if dnf:
                dnf_lap = int(rng.integers(5, total_laps))
                series = series[:dnf_lap]
                total_time = float(np.sum(series))

            pts = float(points[finish_pos - 1]) if finish_pos <= len(points) and not dnf else 0.0
            results.append({
                "id": res_id,
                "race_id": rid,
                "driver_id": e["driver_id"],
                "car_id": e["car_id"],
                "start_position": int(e["start_position"]),
                "finish_position": finish_pos,
                "status": status,
                "points": pts,
                "total_time_sec": round(total_time, 3),
                "fastest_lap_sec": round(fastest, 3),
            })
            res_id += 1

            for lap_num, lap_t in enumerate(series, start=1):
                laps.append({
                    "id": lap_id,
                    "race_id": rid,
                    "driver_id": e["driver_id"],
                    "lap_number": lap_num,
                    "lap_time_sec": round(float(lap_t), 3),
                })
                lap_id += 1

            max_pit = 3 if weather_map[rid]["condition"] in ("Light Rain", "Heavy Rain") else 2
            pit_count = int(rng.integers(1, max_pit + 1))
            if len(series) > pit_count + 2:
                pit_laps = sorted(rng.choice(np.arange(3, len(series) - 1), size=pit_count, replace=False).tolist())
            else:
                pit_laps = []
            for stop_num, pit_lap in enumerate(pit_laps, start=1):
                duration = float(rng.normal(2.6, 0.35))
                if weather_map[rid]["condition"] in ("Light Rain", "Heavy Rain"):
                    duration += float(rng.uniform(0.2, 0.8))
                pits.append({
                    "id": pit_id,
                    "race_id": rid,
                    "driver_id": e["driver_id"],
                    "stop_number": stop_num,
                    "lap_number": int(pit_lap),
                    "duration_sec": round(max(1.7, duration), 3),
                })
                pit_id += 1

    results_df = pd.DataFrame(results)
    laps_df = pd.DataFrame(laps)
    pits_df = pd.DataFrame(pits)

    conn = sqlite3.connect(str(db))
    try:
        create_schema(conn)
        teams_df.to_sql("teams", conn, if_exists="append", index=False)
        drivers_df.to_sql("drivers", conn, if_exists="append", index=False)
        cars_df.to_sql("cars", conn, if_exists="append", index=False)
        races_df.to_sql("races", conn, if_exists="append", index=False)
        weather_df.to_sql("weather", conn, if_exists="append", index=False)
        results_df.to_sql("race_results", conn, if_exists="append", index=False)
        laps_df.to_sql("lap_times", conn, if_exists="append", index=False)
        pits_df.to_sql("pit_stops", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()

    print(f"Seed complete: {db}")
    print(
        f"Rows -> teams={len(teams_df)}, drivers={len(drivers_df)}, cars={len(cars_df)}, "
        f"races={len(races_df)}, weather={len(weather_df)}, results={len(results_df)}, "
        f"laps={len(laps_df)}, pits={len(pits_df)}"
    )


if __name__ == "__main__":
    main()
