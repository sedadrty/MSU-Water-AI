"""Jar-test AI prototype: group-aware validation and near-optimal alum selection.

Install: python -m pip install pandas numpy scikit-learn joblib
Train:   python train_model.py --data data.csv
Predict: python train_model.py --predict --ph 7.3 --turbidity 373 --temperature 30

Research prototype only. Verify recommendations with jar tests before use.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

FEATURES = ["pH", "Turbidity", "Temperature", "Alum"]
REQUIRED = ["Test_ID", *FEATURES, "Final_Turbidity"]
MODEL_FILE = "water_ai_model.joblib"
# Examples only: adjust these criteria with your advisor/plant operator.
DEFAULT_TARGET_NTU = 5.0
DEFAULT_TOLERANCE_NTU = 1.0
DEFAULT_NEAR_BEST_NTU = 1.0  # allowed increase over best final turbidity
DEFAULT_COST_WEIGHT = 0.35


def load_data(path):
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    for col in FEATURES + ["Final_Turbidity"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[REQUIRED].isna().any().any():
        raise ValueError("CSV has missing values")
    if (df[["Turbidity", "Alum", "Final_Turbidity"]] < 0).any().any():
        raise ValueError("Turbidity, Alum and Final_Turbidity must be nonnegative")
    if (~df["pH"].between(0, 14)).any():
        raise ValueError("pH must be between 0 and 14")
    # Verify conditions do not vary within a Jar Test group.
    for name in ["pH", "Turbidity", "Temperature"]:
        if (df.groupby("Test_ID")[name].nunique() > 1).any():
            raise ValueError(f"{name} differs within a Test_ID; inspect dataset")
    return df


def make_model():
    return RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )


def validate_by_jar_test(df):
    groups = df["Test_ID"]
    if groups.nunique() < 3:
        raise ValueError("Need at least 3 independent Jar Test groups")
    splitter = LeaveOneGroupOut()
    actual, predicted = [], []
    per_group = []
    for train_idx, test_idx in splitter.split(df[FEATURES], df["Final_Turbidity"], groups):
        model = make_model()
        model.fit(df.iloc[train_idx][FEATURES], df.iloc[train_idx]["Final_Turbidity"])
        y = df.iloc[test_idx]["Final_Turbidity"].to_numpy()
        p = model.predict(df.iloc[test_idx][FEATURES])
        actual.extend(y.tolist())
        predicted.extend(p.tolist())
        per_group.append({
            "Test_ID": str(df.iloc[test_idx]["Test_ID"].iloc[0]),
            "MAE_NTU": round(float(mean_absolute_error(y, p)), 3),
        })
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted))
    return {"MAE_NTU": round(mae, 4), "RMSE_NTU": round(rmse, 4),
            "R2": round(r2, 4), "by_test": per_group}


def train(csv_path, model_path):
    df = load_data(csv_path)
    metrics = validate_by_jar_test(df)
    print("Independent Jar Test groups:", df["Test_ID"].nunique())
    print("Leave-One-Jar-Test-Out validation:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if df["Test_ID"].nunique() < 10:
        print("WARNING: Few independent Jar Tests; predictions are exploratory only.")
    model = make_model()
    model.fit(df[FEATURES], df["Final_Turbidity"])
    group_conditions = df.groupby("Test_ID", sort=False).first().reset_index()
    group_doses = df.groupby("Test_ID")["Alum"].agg(["min", "max"])
    bundle = {
        "model": model, "features": FEATURES, "validation": metrics,
        "conditions": group_conditions[["Test_ID", "pH", "Turbidity", "Temperature"]],
        "group_doses": group_doses,
        "jar_observations": df[REQUIRED].copy(),
        "ranges": {k: (float(df[k].min()), float(df[k].max())) for k in FEATURES[:3]},
        "chemistry": "Assumes pure Al2(SO4)3Â·18H2O and Ca(OH)2; not a plant lime setpoint",
    }
    joblib.dump(bundle, model_path)
    print(f"Saved model: {model_path}")
    return bundle


def theoretical_lime_mg_l(alum_mg_l, alum_purity=1.0, lime_purity=1.0):
    """Stoichiometric mass for Al2(SO4)3Â·18H2O + 3 Ca(OH)2.

    alum_mg_l is the mass of commercial alum product; purity is the mass
    fraction of Al2(SO4)3Â·18H2O in that product. Lime result is commercial
    Ca(OH)2 product required for full stoichiometric neutralization.
    This is NOT the actual lime dose needed for raw water with alkalinity.
    """
    if not (0 < alum_purity <= 1 and 0 < lime_purity <= 1):
        raise ValueError("Purities must be fractions in (0, 1]")
    if alum_mg_l < 0:
        raise ValueError("Alum dose must be nonnegative")
    mw_alum_18h2o = 666.42
    mw_caoh2 = 74.093
    return float(alum_mg_l * alum_purity * (3 * mw_caoh2 / mw_alum_18h2o) / lime_purity)


def recommend(bundle, ph, turbidity, temperature, target_ntu=5.0,
              tolerance_ntu=1.0, alum_purity=1.0, lime_purity=1.0,
              near_best_ntu=DEFAULT_NEAR_BEST_NTU):
    """Near-optimal dose: lowest dose within 1 NTU of the best result.

    Exact known raw-water conditions: use measured Jar Test results.
    Unseen conditions: use model predictions within nearest Jar Test dose range.
    Experimental upper limit is target_ntu + tolerance_ntu (default 6 NTU).
    """
    vals = [ph, turbidity, temperature, target_ntu, tolerance_ntu, near_best_ntu]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("Inputs must be finite numbers")
    if not (0 <= ph <= 14 and turbidity >= 0 and target_ntu > 0
            and tolerance_ntu >= 0 and near_best_ntu >= 0):
        raise ValueError("Invalid input or criteria")
    values = {"pH": ph, "Turbidity": turbidity, "Temperature": temperature}
    outside = [k for k, val in values.items()
               if not bundle["ranges"][k][0] <= val <= bundle["ranges"][k][1]]
    if outside:
        return {"status": "OUT_OF_TRAINING_RANGE", "outside_features": outside,
                "message": "No recommendation; perform a new Jar Test."}

    conditions = bundle["conditions"]
    # Exact matches use measured data, not smoothed model predictions.
    exact = np.ones(len(conditions), dtype=bool)
    for k, val in values.items():
        exact &= np.isclose(conditions[k].to_numpy(dtype=float), val, atol=1e-6, rtol=0)
    if exact.any():
        closest = conditions.loc[exact, "Test_ID"].iloc[0]
        if "jar_observations" not in bundle:
            return {"status": "RETRAIN_REQUIRED", "message": "Run python train_model.py to save Jar Test observations."}
        observations = bundle["jar_observations"]
        rows = observations[observations["Test_ID"] == closest].copy()
        rows = rows.groupby("Alum", as_index=False)["Final_Turbidity"].mean().sort_values("Alum")
        doses = rows["Alum"].to_numpy(dtype=float)
        turbidities = rows["Final_Turbidity"].to_numpy(dtype=float)
        source = "MEASURED_JAR_TEST"
    else:
        scales = {k: max(hi - lo, 1e-6) for k, (lo, hi) in bundle["ranges"].items()}
        distance = sum(((conditions[k] - values[k]) / scales[k]) ** 2 for k in values)
        closest = conditions.loc[distance.idxmin(), "Test_ID"]
        dose_limits = bundle["group_doses"].loc[closest]
        doses = np.arange(int(np.ceil(dose_limits["min"])),
                          int(np.floor(dose_limits["max"])) + 1, dtype=float)
        if not len(doses):
            return {"status": "NO_CANDIDATES"}
        X = pd.DataFrame({"pH": ph, "Turbidity": turbidity,
                          "Temperature": temperature, "Alum": doses})
        turbidities = bundle["model"].predict(X[FEATURES]).astype(float)
        source = "AI_PREDICTION"

    ceiling = target_ntu + tolerance_ntu
    valid = np.isfinite(turbidities) & (turbidities >= 0) & (turbidities <= ceiling)
    if not valid.any():
        return {"status": "NO_FEASIBLE_DOSE", "maximum_acceptable_NTU": ceiling,
                "message": "No tested/predicted dose meets the ceiling; repeat Jar Test."}

    best_idx = int(np.flatnonzero(valid)[np.argmin(turbidities[valid])])
    best_ntu = float(turbidities[best_idx])
    near_limit = min(ceiling, best_ntu + near_best_ntu)
    near = valid & (turbidities <= near_limit + 1e-10)
    # Exclude dominated candidates: a higher dose with no better turbidity.
    efficient = near.copy()
    for i in np.flatnonzero(near):
        if np.any(near & (doses < doses[i]) & (turbidities <= turbidities[i])):
            efficient[i] = False
    selected_idx = int(np.flatnonzero(efficient)[np.argmin(doses[efficient])])
    selected = float(doses[selected_idx])
    best_dose = float(doses[best_idx])

    def compact(i):
        return {"alum_mg_L": float(doses[i]),
                "final_turbidity_NTU": round(float(turbidities[i]), 3)}

    # For measured data, report only tested doses; for predictions, sampled 1 mg/L.
    nearby = [compact(i) for i in range(len(doses))
              if abs(doses[i] - selected) <= 6]
    return {
        "status": "EXPLORATORY_ONLY",
        "data_source": source,
        "alum_dose_mg_L": selected,
        "final_turbidity_NTU": round(float(turbidities[selected_idx]), 3),
        "theoretical_lime_CaOH2_mg_L": round(theoretical_lime_mg_l(selected, alum_purity, lime_purity), 3),
        "lowest_turbidity_dose": compact(best_idx),
        "near_optimal_dose_range_mg_L": [float(doses[efficient].min()), float(doses[efficient].max())],
        "near_best_allowance_NTU": near_best_ntu,
        "near_best_limit_NTU": round(near_limit, 3),
        "quality_target_NTU": target_ntu,
        "maximum_acceptable_NTU": ceiling,
        "alum_saved_vs_best_mg_L": round(best_dose - selected, 3),
        "comparison_near_selected": nearby,
        "nearest_jar_test_id": str(closest),
        "warning": "Research prototype only. Validate with new Jar Tests. "
                   "Lime is theoretical stoichiometry, not an actual plant setpoint. "
                   "The 6 NTU ceiling is a project criterion, not a drinking-water standard."
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--model", default=MODEL_FILE)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--ph", type=float)
    parser.add_argument("--turbidity", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--target-ntu", type=float, default=DEFAULT_TARGET_NTU)
    parser.add_argument("--tolerance-ntu", type=float, default=DEFAULT_TOLERANCE_NTU)
    parser.add_argument("--alum-purity", type=float, default=1.0)
    parser.add_argument("--lime-purity", type=float, default=1.0)
    parser.add_argument("--near-best-ntu", type=float, default=DEFAULT_NEAR_BEST_NTU)
    args = parser.parse_args()
    if args.predict:
        for name in ["ph", "turbidity", "temperature"]:
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} required for prediction")
        bundle = joblib.load(args.model)  # Load only a model file you created/trust.
        result = recommend(bundle, args.ph, args.turbidity, args.temperature,
                           args.target_ntu, args.tolerance_ntu, args.alum_purity, args.lime_purity,
                           args.near_best_ntu)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        train(args.data, args.model)

if __name__ == "__main__":
    main()