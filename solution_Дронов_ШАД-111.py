# -*- coding: utf-8 -*-
"""
Полусеместровый контроль №4 — Поиск аномальных респондентов в активности SoS.

Запуск:
    python solution_FIO_GROUP.py
    python solution_FIO_GROUP.py --data path/to/data_train --out output

После запуска создаётся папка output/ с обязательными файлами:
    output/anomalies.csv
    output/anomaly_reasons.csv
    output/plots/total_ots_before_after.png
    output/plots/category_ots_change.png
    output/plots/daily_anomaly_count.png

КРАТКОЕ ОПИСАНИЕ АЛГОРИТМА (подробнее — в README.md):
  Единица анализа (триггер): (SubjectID, researchdate, BrandID, CategoryDelivery).
  Единица удаления:          (SubjectID, researchdate).
  daily_ots(i, j, k) = Weight(i, k) * count_rows(i, j, k), где Weight(i, k) —
  дневной вес респондента (медиана веса в пределах респондент-дня, т.к. в данных
  встречается небольшой разброс), count_rows — число строк респондента по этому
  бренду/категории за день (только BrandinDelivery == 1 и непустой CategoryDelivery).

  Аномалия = чрезмерно ВЫСОКИЙ daily_ots относительно ИСТОРИЧЕСКОГО уровня самого
  бренда. Для каждого бренда строится эталонное распределение log(daily_ots) по всем
  его (респондент, день) наблюдениям за период. Сила аномалии:
      score = 0.6745 * (L - median_L) / MAD_L           (робастный modified z-score,
                                                          Iglewicz & Hoaglin, 1993)
  где L = log1p(daily_ots). Лог-пространство выбрано, т.к. OTS мультипликативен и
  сильно скошен; в логах внутрибрендовое распределение приближённо симметрично,
  что и требуется для modified z-score.

  Порог threshold считается ИЗ ДАННЫХ и свой для каждого бренда:
      threshold = max( z(Q3 + 3*IQR),  3.5 )
  — это "дальняя" ограда Тьюки (экстремальный выброс, k=3), пересчитанная в те же
  z-единицы, но не ниже классического порога 3.5. Решение: score > threshold.

  Защита малых выборок / редких брендов: если у бренда < N_MIN наблюдений, эталон
  берётся на уровне CategoryDelivery (иерархический фолбэк). Если MAD == 0 — фолбэк
  на категориальный/глобальный MAD.

  Защита от удаления малого OTS: дополнительно требуется daily_ots >= глобальной
  медианы daily_ots (абсолютный пол). Малый OTS не может стать причиной удаления.

  Никаких захардкоженных SubjectID / дат / брендов / целевого процента удалений.
  Все пороги — стандартные статистические константы либо квантили самих данных,
  поэтому алгоритм воспроизводим и устойчив к смене периода/категорий/брендов.
"""

import argparse
import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Параметры алгоритма ---------------------------
N_MIN = 30        # мин. число наблюдений бренда для бренд-уровневого эталона
MODZ_FLOOR = 3.5  # классический порог modified z-score (Iglewicz & Hoaglin)
TUKEY_K = 3.0     # k для "дальней" ограды Тьюки (экстремальный выброс)
# random_state не требуется: в алгоритме нет случайности (медианы/квантили
# детерминированы), поэтому повторный запуск даёт тот же anomalies.csv.


# ----------------------------- Загрузка данных -------------------------------
def find_data_dir(user_path):
    """Находит каталог с parquet-данными. Без хардкода конкретных файлов."""
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates += [
        "data_train", os.path.join("data", "data_train"),
        os.path.join("data", "data_train", "data_train"), "data", ".",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            if glob.glob(os.path.join(c, "**", "*.parquet"), recursive=True):
                return c
    raise FileNotFoundError(
        "Не найдены parquet-файлы. Укажите путь: --data path/to/data_train"
    )


def load_data(data_dir):
    """Читает все parquet рекурсивно и собирает в один DataFrame.

    Файлы партиционированы по месяцам; диапазоны researchdate не пересекаются,
    поэтому простая конкатенация не создаёт дублей.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"),
                             recursive=True))
    # игнорируем служебный пример с эталонными аномалиями, если он рядом
    files = [f for f in files if "invalidResp" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError("Не найдены parquet-файлы в " + data_dir)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Унификация имени столбца категории поставки.
    if "CategoryDelivery" not in df.columns:
        if "CategoryNameDelivery" in df.columns:
            df["CategoryDelivery"] = df["CategoryNameDelivery"]
        else:
            raise KeyError("Нет столбца CategoryDelivery / CategoryNameDelivery")

    # decimal -> float
    for c in ["Weight", "week_weight", "month_weight"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    df["researchdate"] = pd.to_datetime(df["researchdate"]).dt.date
    return df


# ----------------------------- Агрегация OTS ---------------------------------
def compute_daily_ots(df):
    """Считает daily_ots на уровне триггера (Subject, date, CategoryDelivery, BrandID).

    Берутся только строки поставки: BrandinDelivery == 1 и непустой CategoryDelivery.
    Weight(i, k) — дневной вес респондента (медиана веса в пределах респондент-дня).
    """
    mask = (df["BrandinDelivery"] == 1)
    cat = df["CategoryDelivery"].astype("string").str.strip()
    mask &= cat.notna() & (cat != "")
    d1 = df.loc[mask].copy()

    # Дневной вес респондента Weight(i, k).
    dweight = (d1.groupby(["SubjectID", "researchdate"])["Weight"]
                 .median().rename("dweight").reset_index())

    # count_rows(i, j, k)
    cnt = (d1.groupby(["SubjectID", "researchdate",
                       "CategoryDelivery", "BrandID"])
             .size().rename("count_rows").reset_index())

    cnt = cnt.merge(dweight, on=["SubjectID", "researchdate"], how="left")
    cnt["daily_ots"] = cnt["dweight"] * cnt["count_rows"]

    # Название бренда (детерминированно, первое непустое) для диагностики.
    bname = (d1.dropna(subset=["Brand"]).groupby("BrandID")["Brand"]
               .first().rename("Brand").reset_index())
    cnt = cnt.merge(bname, on="BrandID", how="left")
    cnt["Brand"] = cnt["Brand"].fillna("")
    return cnt


# ----------------------------- Поиск аномалий --------------------------------
def _robust_group_stats(cnt, keys, value="L"):
    """median, MAD, Q1, Q3, n по группе keys (transform -> массивы по строкам)."""
    g = cnt.groupby(keys)[value]
    med = g.transform("median").to_numpy()
    mad = g.transform(lambda x: (x - x.median()).abs().median()).to_numpy()
    q1 = g.transform(lambda x: x.quantile(0.25)).to_numpy()
    q3 = g.transform(lambda x: x.quantile(0.75)).to_numpy()
    n = g.transform("size").to_numpy()
    return med, mad, q1, q3, n


def detect_anomalies(cnt):
    """Помечает аномальные триггеры. Возвращает cnt с колонками score/threshold/flag."""
    cnt = cnt.copy()
    cnt["L"] = np.log1p(cnt["daily_ots"].to_numpy())

    # Эталон уровня бренда (CategoryDelivery, BrandID) и уровня категории.
    bmed, bmad, bq1, bq3, bn = _robust_group_stats(
        cnt, ["CategoryDelivery", "BrandID"])
    cmed, cmad, cq1, cq3, _ = _robust_group_stats(cnt, ["CategoryDelivery"])

    use_brand = bn >= N_MIN  # иерархический фолбэк для редких брендов
    med = np.where(use_brand, bmed, cmed)
    mad = np.where(use_brand, bmad, cmad)
    q1 = np.where(use_brand, bq1, cq1)
    q3 = np.where(use_brand, bq3, cq3)

    # Защита от MAD == 0 (вырожденные распределения, напр. count_rows==1 у всех).
    global_mad = max(float(np.median(np.abs(cnt["L"] - cnt["L"].median()))), 1e-6)
    mad = np.where(mad <= 1e-9,
                   np.where(cmad > 1e-9, cmad, global_mad), mad)

    iqr = np.maximum(q3 - q1, 1e-9)

    # score — робастный modified z-score в лог-пространстве.
    score = 0.6745 * (cnt["L"].to_numpy() - med) / mad

    # threshold — "дальняя" ограда Тьюки в z-единицах, но не ниже 3.5.
    far_L = q3 + TUKEY_K * iqr
    threshold = np.maximum(0.6745 * (far_L - med) / mad, MODZ_FLOOR)

    # Абсолютный пол: малый OTS не может быть аномалией.
    ots_floor = float(cnt["daily_ots"].median())

    cnt["score"] = np.round(score, 4)
    cnt["threshold"] = np.round(threshold, 4)
    cnt["ref_level"] = np.where(use_brand, "brand", "category")
    cnt["ref_n"] = bn.astype(int)
    cnt["flag"] = (score > threshold) & (cnt["daily_ots"].to_numpy() >= ots_floor)
    cnt.attrs["ots_floor"] = ots_floor
    return cnt


def build_reasons(cnt_flagged):
    """Формирует anomaly_reasons по помеченным триггерам."""
    f = cnt_flagged.loc[cnt_flagged["flag"]].copy()

    def _reason(r):
        return (f"daily_ots={r.daily_ots:,.0f} превышает '{r.ref_level}'-эталон бренда: "
                f"робастный z={r.score} > порог {r.threshold} "
                f"(дальняя ограда Тьюки Q3+3·IQR, ≥3.5); "
                f"count_rows={int(r.count_rows)}; выше глобального пола OTS")

    f["reason"] = f.apply(_reason, axis=1)
    cols = ["SubjectID", "researchdate", "BrandID", "Brand",
            "CategoryDelivery", "daily_ots", "score", "threshold", "reason"]
    f = f[cols].sort_values(["researchdate", "SubjectID", "daily_ots"],
                            ascending=[True, True, False]).reset_index(drop=True)
    f["daily_ots"] = f["daily_ots"].round(3)
    return f


# ----------------------------- Обязательные графики --------------------------
def plot_total_ots_before_after(cnt, removed_pairs, path):
    """Суммарный дневной OTS до и после удаления аномалий."""
    idx = cnt.set_index(["SubjectID", "researchdate"]).index
    is_removed = idx.isin(removed_pairs)
    before = cnt.groupby("researchdate")["daily_ots"].sum()
    after = (cnt.loc[~is_removed].groupby("researchdate")["daily_ots"].sum()
               .reindex(before.index, fill_value=0.0))
    b, a = before / 1000.0, after / 1000.0  # в тыс.
    ab, aa = b.mean(), a.mean()
    pct = 100 * aa / ab if ab else 0.0

    fig, ax = plt.subplots(figsize=(15, 7))
    x = list(range(len(before)))
    ax.plot(x, b.values, color="red", label="OTS_before")
    ax.plot(x, a.values, color="green", label="OTS_after")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%m-%d") for d in before.index],
                       rotation=90, fontsize=6)
    ax.set_xlabel("Дата"); ax.set_ylabel("OTS (в тыс.)")
    ax.set_title("Изменение суммарного OTS по дням (до/после удаления аномалий)\n"
                 f"avg_before={ab:,.2f}, avg_after={aa:,.2f}, сохранено OTS = {pct:.2f}%")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_category_ots_change(cnt, removed_pairs, path):
    """Гистограмма изменения OTS по CategoryDelivery в процентах."""
    idx = cnt.set_index(["SubjectID", "researchdate"]).index
    is_removed = idx.isin(removed_pairs)
    before = cnt.groupby("CategoryDelivery")["daily_ots"].sum()
    after = (cnt.loc[~is_removed].groupby("CategoryDelivery")["daily_ots"].sum()
               .reindex(before.index, fill_value=0.0))
    change = (after - before) / before * 100.0
    change = change.sort_index()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(range(len(change)), change.values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(change)))
    ax.set_xticklabels(change.index, rotation=90, fontsize=7)
    ax.set_xlabel("Категория"); ax.set_ylabel("%")
    ax.set_title("Гистограмма изменения суммарного OTS по категориям, % (после − до)")
    for i, v in enumerate(change.values):
        ax.annotate(f"{v:.1f}", (i, v), ha="center",
                    va="top" if v < 0 else "bottom", fontsize=6)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_daily_anomaly_count(reasons, path):
    """Столбчатый график числа аномальных респондентов по дням."""
    pairs = reasons[["SubjectID", "researchdate"]].drop_duplicates()
    per_day = pairs.groupby("researchdate").size()
    all_days = pd.Series(0, index=sorted(reasons["researchdate"].unique())) \
        if len(reasons) else pd.Series(dtype=int)
    per_day = per_day.reindex(per_day.index.union(all_days.index), fill_value=0) \
        if len(per_day) else per_day
    total = len(pairs); uniq = pairs["SubjectID"].nunique()

    fig, ax = plt.subplots(figsize=(16, 6))
    if len(per_day):
        ax.bar(range(len(per_day)), per_day.values)
        ax.set_xticks(range(len(per_day)))
        ax.set_xticklabels([d.strftime("%m-%d") for d in per_day.index],
                           rotation=90, fontsize=6)
    ax.set_xlabel("Дата"); ax.set_ylabel("Количество аномальных респондентов")
    ax.set_title(f"Гистограмма количества удалённых респондентов. "
                 f"Всего удалено {total}, из них уникальных {uniq}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------- Аналитические возможности (п. 8.2) ------------------
def before_after_by_dimension(df, removed_pairs, dim_col, out_png=None, cnt=None):
    """OTS до/после очистки в разрезе любого признака респондента/ресурса/категории.

    dim_col — напр. 'Пол', 'Возраст', 'Регион', 'Федеральный_округ',
    'ResourceName', 'ResourceType', 'Platform', 'UseType',
    'CategoryDelivery', 'Category1', 'Category2', 'Category3'.
    Возвращает DataFrame [dim, ots_before, ots_after, change_pct].
    cnt можно передать заранее посчитанным (ускорение).
    """
    if cnt is None:
        cnt = compute_daily_ots(df)
    key = ["SubjectID", "researchdate", "CategoryDelivery", "BrandID"]
    if dim_col in cnt.columns:
        # признак уже на уровне триггера (CategoryDelivery, Brand) — берём как есть
        cnt = cnt.copy()
        cnt[dim_col] = cnt[dim_col].fillna("—")
    else:
        src = df.loc[(df["BrandinDelivery"] == 1)].copy()
        if dim_col not in src.columns:
            raise KeyError(f"Нет столбца {dim_col}")
        # детерминированный представитель признака по триггеру (первое непустое).
        dim = (src.dropna(subset=[dim_col]).groupby(key)[dim_col]
                  .first().rename(dim_col).reset_index())
        cnt = cnt.merge(dim, on=key, how="left")
        cnt[dim_col] = cnt[dim_col].fillna("—")

    idx = cnt.set_index(["SubjectID", "researchdate"]).index
    is_removed = idx.isin(removed_pairs)
    before = cnt.groupby(dim_col)["daily_ots"].sum()
    after = (cnt.loc[~is_removed].groupby(dim_col)["daily_ots"].sum()
               .reindex(before.index, fill_value=0.0))
    res = pd.DataFrame({"ots_before": before, "ots_after": after})
    res["change_pct"] = (res["ots_after"] - res["ots_before"]) / res["ots_before"] * 100
    res = res.sort_values("ots_before", ascending=False)

    if out_png:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(range(len(res)), res["change_pct"].values)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(res)))
        ax.set_xticklabels(res.index, rotation=90, fontsize=7)
        ax.set_ylabel("% изменения OTS"); ax.set_title(f"OTS до/после по '{dim_col}'")
        fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return res


def querytext_table(df, subject_id, date):
    """Таблица поисковых запросов QueryText для выбранного респондента и дня."""
    if not isinstance(date, (pd.Timestamp,)) and not hasattr(date, "year"):
        date = pd.to_datetime(date).date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    m = (df["SubjectID"] == subject_id) & (df["researchdate"] == date)
    cols = [c for c in ["Start", "QueryText", "Brand", "CategoryDelivery",
                        "ResourceName", "BrandinDelivery"] if c in df.columns]
    return df.loc[m, cols].sort_values("Start") if "Start" in cols else df.loc[m, cols]


def brand_ots_by_day(df, removed_pairs, brand_id, category=None, out_png=None, cnt=None):
    """Изменение OTS по дням для выбранного бренда до и после очистки."""
    if cnt is None:
        cnt = compute_daily_ots(df)
    m = cnt["BrandID"] == brand_id
    if category is not None:
        m &= cnt["CategoryDelivery"] == category
    sub = cnt.loc[m]
    idx = sub.set_index(["SubjectID", "researchdate"]).index
    is_removed = idx.isin(removed_pairs)
    before = sub.groupby("researchdate")["daily_ots"].sum()
    after = (sub.loc[~is_removed].groupby("researchdate")["daily_ots"].sum()
               .reindex(before.index, fill_value=0.0))
    if out_png:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(range(len(before)), before.values, "r-o", ms=3, label="до")
        ax.plot(range(len(before)), after.values, "g-o", ms=3, label="после")
        ax.set_xticks(range(len(before)))
        ax.set_xticklabels([d.strftime("%m-%d") for d in before.index],
                           rotation=90, fontsize=6)
        ax.set_title(f"OTS по дням, бренд {brand_id}"); ax.legend()
        fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return pd.DataFrame({"ots_before": before, "ots_after": after})


# ----------------------------- main ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="каталог с parquet (data_train)")
    ap.add_argument("--out", default="output", help="каталог вывода")
    args = ap.parse_args()

    data_dir = find_data_dir(args.data)
    out_dir = args.out
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[1/5] Чтение данных из: {data_dir}")
    df = load_data(data_dir)
    print(f"      строк: {len(df):,}, респондентов: {df['SubjectID'].nunique():,}, "
          f"дней: {df['researchdate'].nunique()}")

    print("[2/5] Агрегация daily_ots ...")
    cnt = compute_daily_ots(df)
    print(f"      триггеров (Subject,date,Cat,Brand): {len(cnt):,}")

    print("[3/5] Поиск аномалий ...")
    cnt = detect_anomalies(cnt)
    reasons = build_reasons(cnt)
    anomalies = (reasons[["SubjectID", "researchdate"]]
                 .drop_duplicates().sort_values(["researchdate", "SubjectID"])
                 .reset_index(drop=True))
    removed_pairs = pd.MultiIndex.from_frame(
        anomalies[["SubjectID", "researchdate"]])
    print(f"      триггеров-аномалий: {len(reasons)} | "
          f"удаляемых пар (Subject,date): {len(anomalies)} | "
          f"уникальных респондентов: {anomalies['SubjectID'].nunique()}")
    print(f"      доля удалённых респондент-дней: "
          f"{100*len(anomalies)/cnt[['SubjectID','researchdate']].drop_duplicates().shape[0]:.4f}%")

    print("[4/5] Запись CSV ...")
    anomalies.to_csv(os.path.join(out_dir, "anomalies.csv"), index=False)
    reasons.to_csv(os.path.join(out_dir, "anomaly_reasons.csv"), index=False)

    print("[5/5] Построение обязательных графиков ...")
    plot_total_ots_before_after(
        cnt, removed_pairs, os.path.join(plots_dir, "total_ots_before_after.png"))
    plot_category_ots_change(
        cnt, removed_pairs, os.path.join(plots_dir, "category_ots_change.png"))
    plot_daily_anomaly_count(
        reasons, os.path.join(plots_dir, "daily_anomaly_count.png"))

    # --- демонстрация аналитических возможностей (без переписывания функций) ---
    extra = os.path.join(plots_dir, "extra")
    os.makedirs(extra, exist_ok=True)
    for dim in ["Пол", "Возраст", "Регион", "Федеральный_округ",
                "ResourceType", "Platform", "UseType", "CategoryDelivery"]:
        try:
            before_after_by_dimension(
                df, removed_pairs, dim, cnt=cnt,
                out_png=os.path.join(extra, f"by_{dim}.png"))
        except Exception as e:  # noqa
            print(f"      (разрез {dim} пропущен: {e})")
    if len(anomalies):
        s0, d0 = anomalies.iloc[0]["SubjectID"], anomalies.iloc[0]["researchdate"]
        qt = querytext_table(df, s0, d0)
        qt.to_csv(os.path.join(extra, "querytext_example.csv"), index=False)
        b0 = reasons.iloc[0]["BrandID"]
        brand_ots_by_day(df, removed_pairs, b0, cnt=cnt,
                         out_png=os.path.join(extra, "brand_ots_by_day_example.png"))

    print("Готово. Результаты в:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()