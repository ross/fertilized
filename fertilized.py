#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from scipy.optimize import nnls

import numpy as np
import yaml

LBS_TO_GRAMS = 453.592
# Keys in YAML data that are metadata, not nutrients/targets
META_KEYS = {'Rate', 'N', 'available'}
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
        for key, value in data.items():
            if key not in META_KEYS:
                parsed[name].setdefault(key, float(value))
    return parsed


def compute_nutrient_targets(area_sqft, application):
    """Compute absolute nutrient targets in grams for an application.

    Splits N into Ns (slow-release) and Nf (fast-release) based on the
    slow-release fraction specified in the application.
    """
    rate = application.get('Rate', 1)
    n_data = application.get('N')

    if isinstance(n_data, dict):
        n_pct = float(n_data['value'])
        slow_frac = float(n_data.get('slow-release', 0))
    elif n_data is not None:
        n_pct = float(n_data)
        slow_frac = 0.0
    else:
        n_pct = None
        slow_frac = 0.0

    if n_pct is not None:
        # Rate is lbs N per 1000 sqft, derive total product weight from N%
        n_lbs = area_sqft / 1000.0 * rate
        total_product_lbs = n_lbs / (n_pct / 100.0)
        n_grams = n_lbs * LBS_TO_GRAMS
        targets = {
            'Ns': n_grams * slow_frac,
            'Nf': n_grams * (1 - slow_frac),
        }
    else:
        # Rate is lbs product per 1000 sqft (e.g. treatment applications)
        total_product_lbs = area_sqft / 1000.0 * rate
        targets = {'Ns': 0.0, 'Nf': 0.0}

    for key, value in application.items():
        if key in META_KEYS:
            continue
        if isinstance(value, str) and value.endswith('lb'):
            # Absolute weight in lbs per 1000 sqft
            lbs_per_ksqft = float(value[:-2])
            targets[key] = lbs_per_ksqft * (area_sqft / 1000.0) * LBS_TO_GRAMS
        else:
            targets[key] = total_product_lbs * (float(value) / 100.0) * LBS_TO_GRAMS

    return targets


def optimize_fertilizers(targets, fertilizers):
    """Find optimal grams of each fertilizer to meet nutrient targets.

    Uses non-negative least squares which naturally produces sparse solutions.
    Nutrients are discovered dynamically from the union of targets and
    fertilizer profiles.
    """
    fert_names = list(fertilizers.keys())
    n_ferts = len(fert_names)

    # Collect all nutrient keys across targets and fertilizers
    all_nutrients = sorted(
        set(targets.keys())
        | {n for f in fertilizers.values() for n in f.keys()}
    )

    # Build nutrient matrix: each column is a fertilizer, each row is a
    # nutrient. Values are grams of nutrient per gram of fertilizer (pct/100).
    nutrient_matrix = np.zeros((len(all_nutrients), n_ferts))
    for j, name in enumerate(fert_names):
        for i, nutrient in enumerate(all_nutrients):
            nutrient_matrix[i, j] = fertilizers[name].get(nutrient, 0) / 100.0

    target_vec = np.array([targets.get(n, 0) for n in all_nutrients])

    # Only include nutrients that have non-zero targets and can be provided by
    # at least one fertilizer
    active = []
    for i, nutrient in enumerate(all_nutrients):
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
    actuals = {all_nutrients[i]: actual[i] for i in range(len(all_nutrients))}

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
    products = load_yaml('fertilizers.yaml')
    products.update(load_yaml('treatments.yaml'))
    fertilizers = parse_fertilizer_nutrients(products)

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
            all_nutrients = sorted(
                set(targets.keys()) | set(actuals.keys())
            )
            max_label = max(
                (
                    len(n)
                    for n in all_nutrients
                    if n not in ('Ns', 'Nf') and targets.get(n, 0) > 0
                ),
                default=8,
            )
            # +2 for the Ns/Nf indent
            max_label = max(max_label, 4)
            print(
                f"    {'Nutrient':<{max_label}}"
                f" {'Target':>10} {'Actual':>10} {'Error':>8}"
            )
            print(
                f"    {'':-<{max_label}}"
                f" {'':-<10} {'':-<10} {'':-<8}"
            )

            def pct_error(target, actual):
                if target == 0:
                    return ''
                return f"{(actual - target) / target * 100:>+7.1f}%"

            # Show N total with Ns/Nf breakdown first
            n_target = targets.get('Ns', 0) + targets.get('Nf', 0)
            n_actual = actuals.get('Ns', 0) + actuals.get('Nf', 0)
            if n_target > 0 or n_actual > 0:
                print(
                    f"    {'N':<{max_label}} {n_target:>9.1f}g"
                    f" {n_actual:>9.1f}g {pct_error(n_target, n_actual):>8}"
                )
                for nutrient in ('Ns', 'Nf'):
                    t = targets.get(nutrient, 0)
                    a = actuals.get(nutrient, 0)
                    print(
                        f"    {'  ' + nutrient:<{max_label}} {t:>9.1f}g"
                        f" {a:>9.1f}g {pct_error(t, a):>8}"
                    )
            # Show all other nutrients with non-zero targets
            for nutrient in all_nutrients:
                if nutrient in ('Ns', 'Nf'):
                    continue
                t = targets.get(nutrient, 0)
                if t == 0:
                    continue
                a = actuals.get(nutrient, 0)
                print(
                    f"    {nutrient:<{max_label}} {t:>9.1f}g"
                    f" {a:>9.1f}g {pct_error(t, a):>8}"
                )


if __name__ == '__main__':
    main()
