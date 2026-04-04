#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from scipy.optimize import nnls

import numpy as np
import yaml

LBS_TO_GRAMS = 453.592
NUTRIENTS = ['Ns', 'Nf', 'P', 'K', 'Fe']
BASE_DIR = Path(__file__).parent


def load_yaml(filename):
    with open(BASE_DIR / filename) as f:
        return yaml.safe_load(f)


def parse_fertilizer_nutrients(fertilizers):
    """Parse fertilizer data into a dict of {name: {nutrient: percentage}}.

    Splits N into Ns (slow-release) and Nf (fast-release) based on the
    slow-release fraction. Skips fertilizers marked as unavailable.
    """
    parsed = {}
    for name, data in fertilizers.items():
        if data.get('available') is False:
            continue
        parsed[name] = {}
        n_data = data.get('N', 0)
        if isinstance(n_data, dict):
            n_total = float(n_data['value'])
            slow_frac = float(n_data.get('slow-release', 0))
        else:
            n_total = float(n_data)
            slow_frac = 0.0
        parsed[name]['Ns'] = n_total * slow_frac
        parsed[name]['Nf'] = n_total * (1 - slow_frac)
        for nutrient in ('P', 'K', 'Fe'):
            value = data.get(nutrient, 0)
            parsed[name][nutrient] = float(value)
    return parsed


def compute_nutrient_targets(area_sqft, application):
    """Compute absolute nutrient targets in grams for an application.

    Splits N into Ns (slow-release) and Nf (fast-release) based on the
    slow-release fraction specified in the application.
    """
    rate = application['Rate']
    n_data = application['N']
    if isinstance(n_data, dict):
        n_pct = float(n_data['value'])
        slow_frac = float(n_data.get('slow-release', 0))
    else:
        n_pct = float(n_data)
        slow_frac = 0.0

    # Total N in lbs
    n_lbs = area_sqft / 1000.0 * rate
    # Total product weight in lbs
    total_product_lbs = n_lbs / (n_pct / 100.0)
    # Total N in grams
    n_grams = total_product_lbs * (n_pct / 100.0) * LBS_TO_GRAMS

    targets = {
        'Ns': n_grams * slow_frac,
        'Nf': n_grams * (1 - slow_frac),
    }
    for nutrient in ('P', 'K', 'Fe'):
        pct = application.get(nutrient, 0)
        targets[nutrient] = total_product_lbs * (float(pct) / 100.0) * LBS_TO_GRAMS

    return targets


def optimize_fertilizers(targets, fertilizers):
    """Find optimal grams of each fertilizer to meet nutrient targets.

    Uses non-negative least squares which naturally produces sparse solutions.
    """
    fert_names = list(fertilizers.keys())
    n_ferts = len(fert_names)

    # Build nutrient matrix: each column is a fertilizer, each row is a
    # nutrient. Values are grams of nutrient per gram of fertilizer (pct/100).
    nutrient_matrix = np.zeros((len(NUTRIENTS), n_ferts))
    for j, name in enumerate(fert_names):
        for i, nutrient in enumerate(NUTRIENTS):
            nutrient_matrix[i, j] = fertilizers[name][nutrient] / 100.0

    target_vec = np.array([targets[n] for n in NUTRIENTS])

    # Only include nutrients that have non-zero targets and can be provided by
    # at least one fertilizer
    active = []
    for i, nutrient in enumerate(NUTRIENTS):
        if target_vec[i] > 0 and np.any(nutrient_matrix[i] > 0):
            active.append(i)
    active_matrix = nutrient_matrix[active]
    active_targets = target_vec[active]

    weights, _ = nnls(active_matrix, active_targets)

    amounts = {}
    for j, name in enumerate(fert_names):
        if weights[j] >= 0.5:
            amounts[name] = weights[j]

    actual = nutrient_matrix @ weights
    actuals = {NUTRIENTS[i]: actual[i] for i in range(len(NUTRIENTS))}

    return amounts, actuals


def main():
    parser = ArgumentParser(description='Calculate fertilizer requirements')
    parser.add_argument(
        '--area',
        action='append',
        dest='areas',
        metavar='AREA',
        help='limit output to the specified area(s), case-insensitive'
        ' substring match (can be repeated)',
    )
    parser.add_argument(
        '--application',
        action='append',
        dest='applications',
        metavar='APPLICATION',
        help='limit output to the specified application period(s),'
        ' case-insensitive substring match (can be repeated)',
    )
    args = parser.parse_args()

    areas = load_yaml('areas.yaml')
    fertilizers = parse_fertilizer_nutrients(load_yaml('fertilizers.yaml'))

    if args.areas:
        filters = [f.lower() for f in args.areas]
        areas = {
            name: data
            for name, data in areas.items()
            if any(f in name.lower() for f in filters)
        }
        if not areas:
            parser.error(
                f"no areas matched: {', '.join(args.areas)}"
            )

    if args.applications:
        app_filters = [f.lower() for f in args.applications]

    for area_name, area_data in areas.items():
        area_sqft = area_data['square-feet']
        print(f"\n{'=' * 60}")
        print(f"{area_name} ({area_sqft} sq ft)")
        print(f"{'=' * 60}")

        applications = area_data['application']
        if args.applications:
            applications = {
                name: data
                for name, data in applications.items()
                if any(f in name.lower() for f in app_filters)
            }

        for period, application in applications.items():
            targets = compute_nutrient_targets(area_sqft, application)
            amounts, actuals = optimize_fertilizers(targets, fertilizers)

            print(f"\n  {period}")
            print(f"  {'-' * 50}")

            if amounts:
                max_name_len = max(len(n) for n in amounts)
                for name, grams in sorted(
                    amounts.items(), key=lambda x: -x[1]
                ):
                    print(f"    {name:<{max_name_len}}  {grams:>8.1f} g")
            else:
                print("    No fertilizers needed")

            print()
            print(f"    {'Nutrient':<10} {'Target':>10} {'Actual':>10}")
            print(f"    {'':-<10} {'':-<10} {'':-<10}")
            # Show N total, then Ns/Nf breakdown, then P, K, Fe
            n_target = targets['Ns'] + targets['Nf']
            n_actual = actuals['Ns'] + actuals['Nf']
            n_flag = ' *' if abs(n_actual - n_target) > 0.5 else ''
            print(
                f"    {'N':<10} {n_target:>9.1f}g {n_actual:>9.1f}g{n_flag}"
            )
            for nutrient in NUTRIENTS:
                t = targets[nutrient]
                a = actuals[nutrient]
                flag = ' *' if abs(a - t) > 0.5 else ''
                label = f"  {nutrient}" if nutrient in ('Ns', 'Nf') else nutrient
                print(
                    f"    {label:<10} {t:>9.1f}g {a:>9.1f}g{flag}"
                )


if __name__ == '__main__':
    main()
