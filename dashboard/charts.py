"""Altair charts for the dashboard. Every function takes a DataFrame and returns a finished chart.

Chain colors can't be told apart by colour alone (red and green fail the colour-blindness check), so a
chart that uses them also names each chain on its axis. Sequential scales use one hue; signed changes
use two hues around a grey midpoint.
"""

import altair as alt
import pandas as pd

INK, MUTED, BEST = "#17242D", "#5F6E78", "#12805C"
QUIET = "#9AA7AE"
RISE, FALL = "#C97132", "#1E5736"
SEQUENTIAL = ["#E4F1EC", "#9CCDBB", "#4E9F88", "#17685A"]
EMPTY = "#EEF1EF"
SURFACE = "#FFFFFF"


def styled(chart):
    fit = alt.AutoSizeParams(type="fit-x", contains="padding")
    return chart.properties(autosize=fit, padding={"left": 6, "right": 14, "top": 8, "bottom": 6}).configure(
        font="Archivo", background=SURFACE,
    ).configure_view(strokeWidth=0).configure_axis(
        labelColor=MUTED, titleColor=MUTED, gridColor="#E8EDEA", domainColor="#C9D2CD", labelFontSize=12, titleFontSize=12,
    ).configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=12, titleFontSize=12)


def chain_scale(colors, names):
    return alt.Scale(domain=[names.get(k, k) for k in colors], range=list(colors.values()))


def _axis(title, **kwargs):
    return alt.Axis(title=title, ticks=False, domain=False, **kwargs)


def _height(rows, step=34, extra=44):
    return rows * step + extra


def scatter(cards, colors, names, money, distance):
    """Basket price against distance; stores beaten on both are faded."""
    c = cards[cards["complete"]].copy()
    c["Chain"] = c["chain"].map(lambda x: names.get(x, x))
    c["dominated"] = [
        bool(((c["basket_eur"] <= r.basket_eur) & (c["distance_km"] <= r.distance_km)
              & ((c["basket_eur"] < r.basket_eur) | (c["distance_km"] < r.distance_km))).any())
        for r in c.itertuples()
    ]
    c["Basket"] = c["basket_eur"].map(money)
    c["Distance"] = c["distance_km"].map(distance)
    base = alt.Chart(c).encode(
        x=alt.X("distance_km:Q", title="Distance to the store (km)", scale=alt.Scale(zero=True)),
        y=alt.Y("basket_eur:Q", title="Basket price (€)", scale=alt.Scale(zero=False)),
        color=alt.Color("Chain:N", scale=chain_scale(colors, names), legend=None),
        opacity=alt.condition("datum.dominated", alt.value(0.28), alt.value(1)),
        tooltip=["Chain", "Basket", "Distance"],
    )
    labels = base.mark_text(align="left", dx=11, fontSize=12, fontWeight=700, color=INK).encode(text="Chain", color=alt.value(INK))
    return styled((base.mark_circle(size=230) + labels).properties(height=280))

def _unwrap_scalar(val, default=None):
    while isinstance(val, (pd.DataFrame, pd.Series, list, tuple)):
        if isinstance(val, pd.DataFrame):
            if val.empty:
                return default
            val = val.iloc[0, 0]
        elif isinstance(val, pd.Series):
            if val.empty:
                return default
            val = val.iloc[0]
        else:
            if len(val) == 0:
                return default
            val = val[0]
    return default if val is None else val


def _to_float(val, default=0.0):
    scalar = _unwrap_scalar(val, default)
    try:
        num = float(scalar)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if pd.isna(num) else num


def _to_int(val, default=0):
    return int(round(_to_float(val, float(default))))


def worth_going_further(cards, colors, names, cost_per_km, cost_matrix=None):
    """Diverging bar chart of net saving per available item vs. the closest store."""
    if cards is None or not isinstance(cards, pd.DataFrame) or cards.empty:
        return styled(
            alt.Chart(pd.DataFrame({"message": ["No stores available in range"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=160)
        )

    c_df = cards.loc[:, ~cards.columns.duplicated()].copy()
    required = {"chain", "basket_eur", "have", "distance_km"}
    if not required.issubset(c_df.columns):
        return styled(
            alt.Chart(pd.DataFrame({"message": ["Missing basket columns for comparison"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=160)
        )

    have_num = pd.to_numeric(c_df["have"], errors="coerce")
    dist_num = pd.to_numeric(c_df["distance_km"], errors="coerce")
    basket_num = pd.to_numeric(c_df["basket_eur"], errors="coerce")

    valid = c_df[basket_num.notna() & dist_num.notna() & (have_num.fillna(0) > 0)].reset_index(drop=True)
    if valid.empty:
        return styled(
            alt.Chart(pd.DataFrame({"message": ["No basket items available in nearby stores"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=160)
        )

    cm = (
        cost_matrix.loc[:, ~cost_matrix.columns.duplicated()]
        if isinstance(cost_matrix, pd.DataFrame) and not cost_matrix.empty
        else None
    )

    max_have_raw = pd.to_numeric(valid["have"], errors="coerce").max()
    total_basket_items = max(1, int(len(cm)) if cm is not None else _to_int(max_have_raw, 1))

    valid_dist = pd.to_numeric(valid["distance_km"], errors="coerce")
    closest_pos = int(valid_dist.values.argmin())
    closest = valid.iloc[closest_pos]

    closest_chain = str(_unwrap_scalar(closest["chain"], ""))
    closest_dist = _to_float(closest["distance_km"], 0.0)
    closest_basket = _to_float(closest["basket_eur"], 0.0)
    closest_have = max(1, _to_int(closest["have"], 1))
    rate = _to_float(cost_per_km, 0.0)

    rows = []
    for r in valid.itertuples(index=False):
        chain = str(_unwrap_scalar(getattr(r, "chain", ""), ""))
        name = str(names.get(chain, chain))
        dist_km = _to_float(getattr(r, "distance_km", 0.0), 0.0)
        basket_eur = _to_float(getattr(r, "basket_eur", 0.0), 0.0)
        have_n = max(1, _to_int(getattr(r, "have", 1), 1))

        extra_km = max(0.0, (dist_km - closest_dist) * 2.0 * 1.3)
        travel_cost = extra_km * rate

        if cm is not None and chain in cm.columns and closest_chain in cm.columns:
            cols = [closest_chain] if chain == closest_chain else [closest_chain, chain]
            both = cm[cols].dropna()
            if not both.empty and chain != closest_chain:
                s_closest = pd.to_numeric(_unwrap_scalar(both[closest_chain], both[closest_chain]), errors="coerce")
                s_chain = pd.to_numeric(_unwrap_scalar(both[chain], both[chain]), errors="coerce")
                if isinstance(s_closest, pd.DataFrame):
                    s_closest = s_closest.iloc[:, 0]
                if isinstance(s_chain, pd.DataFrame):
                    s_chain = s_chain.iloc[:, 0]
                diff = pd.Series(s_closest) - pd.Series(s_chain)
                gross_saving = _to_float(diff.sum(), 0.0)
                n_comp = max(1, int(diff.notna().sum()))
            else:
                gross_saving = 0.0
                n_comp = have_n
        else:
            ref_avg = closest_basket / closest_have
            cur_avg = basket_eur / have_n
            gross_saving = (ref_avg - cur_avg) * have_n
            n_comp = have_n

        net_saving = gross_saving - travel_cost
        per_item = net_saving / max(1, n_comp)

        is_closest = (chain == closest_chain) and abs(dist_km - closest_dist) < 1e-6
        if is_closest:
            val_label = f"Closest ({have_n}/{total_basket_items} items)"
            plot_val = 0.0
        else:
            sign = "+" if per_item >= 0 else "-"
            val_label = f"{sign}€{abs(per_item):.2f}/item ({have_n}/{total_basket_items} items)"
            plot_val = round(per_item, 3)

        rows.append({
            "Chain": name,
            "net_per_item": plot_val,
            "net_total": round(net_saving, 2),
            "gross_saving": round(gross_saving, 2),
            "travel_cost": round(travel_cost, 2),
            "items_txt": f"{have_n}/{total_basket_items} items",
            "coverage_pct": round(100 * have_n / total_basket_items, 1),
            "distance_km": round(dist_km, 2),
            "verdict": (
                "Closest store" if is_closest
                else ("Worth going further" if per_item > 1e-4 else "Not worth it")
            ),
            "is_closest": is_closest,
        })


    
    c = pd.DataFrame(rows).sort_values("coverage_pct", ascending=False)
    order = c["Chain"].tolist()

    # Keep zero visible and add some breathing room to the horizontal axis.
    lo = min(0.0, float(c["net_per_item"].min()))
    hi = max(0.0, float(c["net_per_item"].max()))
    pad = max(0.08, (hi - lo) * 0.12)

    min_v = lo - pad
    max_v = hi + pad

    x_enc = alt.X(
        "net_per_item:Q",
        title="Net saving per comparable item (€/item)",
        scale=alt.Scale(domain=[min_v, max_v]),
        axis=alt.Axis(format="+.2f", tickCount=6),
    )

    y_enc = alt.Y(
        "coverage_pct:Q",
        title="Basket coverage (%)",
        scale=alt.Scale(domain=[0, 100]),
        axis=alt.Axis(format=".0f", values=[0, 20, 40, 60, 80, 100]),
    )

    zero_rule = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_rule(color=INK, strokeWidth=1.5, strokeDash=[5, 4])
        .encode(x="x:Q")
    )

    points = (
        alt.Chart(c)
        .mark_circle(size=180, opacity=0.9, stroke="white", strokeWidth=1.5)
        .encode(
            x=x_enc,
            y=y_enc,
            color=alt.Color(
                "Chain:N",
                scale=chain_scale(colors, names),
                legend=alt.Legend(title="Supermarket", orient="top", columns=4),
            ),
            tooltip=[
                alt.Tooltip("Chain:N", title="Supermarket"),
                alt.Tooltip("verdict:N", title="Assessment"),
                alt.Tooltip("items_txt:N", title="Available products"),
                alt.Tooltip("coverage_pct:Q", title="Basket coverage (%)", format=".1f"),
                alt.Tooltip("net_per_item:Q", title="Net saving per item (€)", format="+.2f"),
                alt.Tooltip("net_total:Q", title="Net saving total (€)", format="+.2f"),
                alt.Tooltip("gross_saving:Q", title="Gross product saving (€)", format="+.2f"),
                alt.Tooltip("travel_cost:Q", title="Extra travel cost (€)", format=".2f"),
                alt.Tooltip("distance_km:Q", title="Distance (km)", format=".2f"),
            ],
        )
    )

    chart = (zero_rule + points).properties(
        height=_height(len(order), 38),
    )

    return styled(chart)



def sensitivity(sens, current_km):
    long = pd.concat([
        pd.DataFrame({"cost": sens["cost_per_km"], "Total": sens["best_total"], "Plan": "Best plan", "z": 0}),
        pd.DataFrame({"cost": sens["cost_per_km"], "Total": sens["single_total"], "Plan": "One store only", "z": 1}),
    ])
    domain = ["One store only", "Best plan"]
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        order="z:Q",
        x=alt.X("cost:Q", title="Travel cost (€/km)"),
        y=alt.Y("Total:Q", title="Total cost (€)", scale=alt.Scale(zero=False)),
        color=alt.Color("Plan:N", scale=alt.Scale(domain=domain, range=[QUIET, BEST]), legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("Plan:N", scale=alt.Scale(domain=domain, range=[[7, 5], [1, 0]]), legend=None),
        tooltip=[alt.Tooltip("Plan:N"), alt.Tooltip("cost:Q", title="€/km"), alt.Tooltip("Total:Q", format=".2f", title="Total (€)")],
    )
    now = alt.Chart(pd.DataFrame({"x": [current_km]})).mark_rule(color=INK, strokeDash=[2, 3]).encode(x="x:Q")
    return styled((lines + now).properties(height=280))


def coverage(table, names, categories):
    """Comparable products per category and chain."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)), Category=table["category"].map(categories))
    order = t.groupby("Category")["products"].sum().sort_values(ascending=False).index.tolist()
    base = alt.Chart(t).encode(
        x=alt.X("Chain:N", axis=_axis(None, orient="top", labelAngle=0)),
        y=alt.Y("Category:N", sort=order, axis=_axis(None)),
    )
    cells = base.mark_rect(cornerRadius=4, stroke=SURFACE, strokeWidth=2).encode(
        color=alt.condition("datum.products == 0", alt.value(EMPTY), alt.Color(
            "products:Q", scale=alt.Scale(type="sqrt", range=SEQUENTIAL), legend=alt.Legend(title="Products", orient="right"))),
        tooltip=["Category", "Chain", alt.Tooltip("products:Q", title="Products")],
    )
    text = base.mark_text(fontSize=12).encode(
        text="products:Q", color=alt.condition("datum.products >= 40", alt.value("#FFFFFF"), alt.value(INK)))
    return styled((cells + text).properties(height=_height(len(order), 30)))


def index_dumbbell(table, colors, names):
    """Price index per chain: typical (median) and cheapest product, joined by a line (Mercadona = 100)."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)))
    order = t.sort_values("median")["Chain"].tolist()
    long = t.melt(id_vars="Chain", value_vars=["cheapest", "median"], var_name="m", value_name="Index")
    domain = ["Cheapest product", "Typical product (median)"]
    long["Measure"] = long["m"].map({"cheapest": domain[0], "median": domain[1]})
    y = alt.Y("Chain:N", sort=order, axis=_axis(None))
    x = alt.X("Index:Q", title="Price index (Mercadona = 100)", scale=alt.Scale(zero=False, nice=True))
    ref = alt.Chart(pd.DataFrame({"Index": [100]})).mark_rule(color=QUIET, strokeDash=[4, 4]).encode(x="Index:Q")
    link = alt.Chart(t).mark_rule(color="#C9D2CD", strokeWidth=2).encode(x="cheapest:Q", x2="median:Q", y=y)
    dots = alt.Chart(long).mark_circle(size=140, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
        x=x, y=y,
        color=alt.Color("Measure:N", scale=alt.Scale(domain=domain, range=[BEST, INK]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Chain", "Measure", alt.Tooltip("Index:Q", format=".0f")],
    )
    text = alt.Chart(t).mark_text(align="left", dx=10, fontSize=12, color=MUTED).encode(
        x="median:Q", y=y, text=alt.Text("median:Q", format=".0f"))
    return styled((ref + link + dots + text).properties(height=_height(len(order), 40)))


def category_heatmap(table, names, categories):
    """Price against the cheapest chain, per category. Chains with too little data are left grey."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)), Category=table["category"].map(categories))
    t["Compared"] = t["relative"].map(lambda v: "cheapest" if v <= 1.0005 else f"+{100 * (v - 1):.0f}%")
    t["shown"] = t["Compared"].where(~t["low_data"], "few")
    t["capped"] = t["relative"].clip(upper=2.5)
    order = sorted(t["Category"].unique())
    chains = t.groupby("Chain")["relative"].median().sort_values().index.tolist()
    base = alt.Chart(t).encode(
        x=alt.X("Chain:N", sort=chains, axis=_axis(None, orient="top", labelAngle=0)),
        y=alt.Y("Category:N", sort=order, axis=_axis(None)),
    )
    cells = base.mark_rect(cornerRadius=4, stroke=SURFACE, strokeWidth=2).encode(
        color=alt.condition("datum.low_data", alt.value(EMPTY), alt.Color(
            "capped:Q", scale=alt.Scale(domain=[1, 2.5], range=SEQUENTIAL),
            legend=alt.Legend(title="Times the cheapest", format=".1f", orient="right"))),
        tooltip=["Category", "Chain", alt.Tooltip("Compared:N", title="Against cheapest"), alt.Tooltip("products:Q", title="Products")],
    )
    text = base.mark_text(fontSize=11.5).encode(
        text="shown:N", color=alt.condition("datum.capped >= 1.8 && !datum.low_data", alt.value("#FFFFFF"), alt.value(INK)))
    return styled((cells + text).properties(height=_height(len(order), 30)))


def _chain_bars(table, value, title, colors, names, tooltip, fmt):
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)))
    t["text"] = t[value].map(fmt)
    order = t.sort_values(value, ascending=False)["Chain"].tolist()
    y = alt.Y("Chain:N", sort=order, axis=_axis(None))
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(tickCount=5)), y=y,
        color=alt.Color("Chain:N", scale=chain_scale(colors, names), legend=None), tooltip=tooltip,
    )
    return styled((bars + alt.Chart(t).mark_text(align="left", dx=6, color=INK, fontSize=12).encode(
        x=f"{value}:Q", y=y, text="text:N")).properties(height=_height(len(order))))


def own_brand(table, colors, names):
    return _chain_bars(
        table, "gap_pct", "Own brand saving against national brands (%)", colors, names,
        ["Chain", alt.Tooltip("gap_pct:Q", format=".0f", title="Saving (%)"), alt.Tooltip("types:Q", title="Types compared")],
        lambda v: f"{v:.0f}%")


def own_share(table, colors, names):
    return _chain_bars(
        table, "own_share_pct", "Own brand share of comparable products (%)", colors, names,
        ["Chain", alt.Tooltip("own_share_pct:Q", format=".0f", title="Own brand (%)")], lambda v: f"{v:.0f}%")


def spread(table, categories):
    """Within-chain spread against between-chain spread, per category."""
    t = table.assign(Category=table["category"].map(categories))
    order = t["Category"].tolist()
    long = t.melt(id_vars="Category", value_vars=["within", "between"], var_name="m", value_name="Spread")
    domain = ["Between products of one chain", "Between chains"]
    long["Choice"] = long["m"].map({"within": domain[0], "between": domain[1]})
    x = alt.X("Spread:Q", title="Dearest against cheapest (times)", scale=alt.Scale(type="log", domain=[1, 12]),
              axis=alt.Axis(values=[1, 2, 3, 5, 10]))
    y = alt.Y("Category:N", sort=order, axis=_axis(None))
    link = alt.Chart(t).mark_rule(color="#C9D2CD", strokeWidth=2).encode(x="within:Q", x2="between:Q", y=y)
    dots = alt.Chart(long).mark_circle(size=120, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
        x=x, y=y,
        color=alt.Color("Choice:N", scale=alt.Scale(domain=domain, range=[BEST, QUIET]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Category", "Choice", alt.Tooltip("Spread:Q", format=".1f", title="Times")],
    )
    return styled((link + dots).properties(height=_height(len(order), 26)))


def barcode_gap(pairs):
    """Histogram of BM against Consum on the same barcode."""
    width = 4
    centre = ((pairs["gap_pct"].clip(-40, 40) + width / 2) // width * width).astype(int)
    t = centre.value_counts().rename_axis("gap").reset_index(name="products")
    t["side"] = t["gap"].map(lambda v: "BM dearer" if v > 0 else ("Consum dearer" if v < 0 else "About the same (±2%)"))
    domain = ["Consum dearer", "About the same (±2%)", "BM dearer"]
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=4, width=alt.RelativeBandSize(band=0.9)).encode(
        x=alt.X("gap:O", sort="ascending", title="BM price against Consum (%), capped at ±40",
                scale=alt.Scale(domain=list(range(-40, 41, width))), axis=alt.Axis(labelAngle=0, values=list(range(-40, 41, 20)))),
        y=alt.Y("products:Q", title="Products"),
        color=alt.Color("side:N", scale=alt.Scale(domain=domain, range=[FALL, QUIET, RISE]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("side:N", title="Result"), alt.Tooltip("gap:O", title="Gap (%)"), alt.Tooltip("products:Q", title="Products")],
    )
    return styled(bars.properties(height=280))



def route_dumbbell(table):
    """Compare Price list and exact optimizer costs for every basket."""
    data = table.copy().sort_values("saving_eur", ascending=False).reset_index(drop=True)
    data["basket"] = [f"Basket {i + 1:02d}" for i in range(len(data))]

    y = alt.Y(
        "basket:N",
        sort=alt.SortField(field="saving_eur", order="descending"),
        axis=alt.Axis(title=None, labelLimit=120),
    )

    # A connecting line joins the two total costs for each basket.
    lines = (
        alt.Chart(data)
        .mark_rule(color=QUIET, strokeWidth=2)
        .encode(
            y=y,
            x=alt.X("exact_eur:Q", title="Total basket cost (€)", scale=alt.Scale(zero=False)),
            x2="greedy_eur:Q",
        )
    )

    # Each dot represents one method's total cost.
    points = (
        alt.Chart(data)
        .transform_fold(
            ["greedy_eur", "exact_eur"],
            as_=["method", "cost"],
        )
        .transform_calculate(
            method_label=(
                "datum.method === 'greedy_eur' "
                "? 'Price list' : 'Exact optimizer'"
            ),
            stops=(
                "datum.method === 'greedy_eur' "
                "? datum.greedy_stops : datum.exact_stops"
            ),
        )
        .mark_circle(size=85, opacity=1)
        .encode(
            y=y,
            x=alt.X("cost:Q", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "method_label:N",
                scale=alt.Scale(
                    domain=["Price list", "Exact optimizer"],
                    range=[QUIET, "#1E5736"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("basket:N", title="Basket"),
                alt.Tooltip("method_label:N", title="Method"),
                alt.Tooltip("cost:Q", title="Total cost (€)", format=".2f"),
                alt.Tooltip("saving_eur:Q", title="Saving (€)", format="+.2f"),
                alt.Tooltip("greedy_stops:Q", title="Price list stops"),
                alt.Tooltip("exact_stops:Q", title="Exact optimizer stops"),
            ],
        )
    )

    chart = (lines + points).properties(
        height=max(420, min(780, len(data) * 22)),
    )

    return styled(chart)


def moved_share(summary, names, fall_color="#1E5736", rise_color="#F0883E", same_color="#ABB4AD"):
    """Share of prices that rose, stayed and fell, per chain."""
    t = summary.assign(Chain=summary["chain"].map(lambda x: names.get(x, x)))
    order = t["Chain"].tolist()
    long = t.melt(id_vars="Chain", value_vars=["down_pct", "same_pct", "up_pct"], var_name="m", value_name="Share")
    domain = ["Fell", "Unchanged", "Rose"]
    long["Price"] = long["m"].map({"down_pct": "Fell", "same_pct": "Unchanged", "up_pct": "Rose"})
    long["o"] = long["Price"].map({k: i for i, k in enumerate(domain)})
    bars = alt.Chart(long).mark_bar(height=22, stroke=SURFACE, strokeWidth=2).encode(
        x=alt.X("Share:Q", title="Share of products (%)", stack="zero", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(tickCount=5)),
        y=alt.Y("Chain:N", sort=order, axis=_axis(None)), order="o:Q",
        color=alt.Color("Price:N", scale=alt.Scale(domain=domain, range=[fall_color, same_color, rise_color]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Chain", "Price", alt.Tooltip("Share:Q", format=".1f", title="Share (%)")],
    )
    return styled(bars.properties(height=_height(len(order), 40)))


def moves(table, names, color, rounded_left=False):
    """Largest price changes, as bars from zero."""
    t = table.assign(Chain=table["supermarket"].map(lambda x: names.get(x, x)))
    t["Product"] = t["Chain"] + ": " + t["name"].str.slice(0, 34)
    # Invisible suffix: two products cut to the same label must stay two bars, not stack into one
    t["Product"] = [label + "\u200b" * i for i, label in enumerate(t["Product"])]
    t["text"] = t["change_pct"].map(lambda v: f"{v:+.0f}%")
    t["Was"] = t["old_price"].map(lambda v: f"€{v:.2f}")
    t["Now"] = t["new_price"].map(lambda v: f"€{v:.2f}")
    radius = (
        {"cornerRadiusTopLeft": 4, "cornerRadiusBottomLeft": 4}
        if rounded_left
        else {"cornerRadiusTopRight": 4, "cornerRadiusBottomRight": 4}
    )
    y = alt.Y("Product:N", sort=t["Product"].tolist(), axis=_axis(None, labelLimit=225, minExtent=225))
    bars = alt.Chart(t).mark_bar(height=16, color=color, **radius).encode(
        x=alt.X("change_pct:Q", title="Price change (%)", axis=alt.Axis(tickCount=5)), y=y,
        tooltip=["Product", "Was", "Now", alt.Tooltip("change_pct:Q", format="+.1f", title="Change (%)")],
    )
    return styled(bars.properties(height=_height(len(t), 28)))


def category_change(table, names, categories):
    """Average price change per category and chain; only cells with enough products exist."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)), Category=table["category"].map(categories))
    t["text"] = t["average_pct"].map(lambda v: "0.0%" if abs(v) < 0.05 else f"{v:+.1f}%")
    limit = max(5, t["average_pct"].abs().max())
    base = alt.Chart(t).encode(
        x=alt.X("Chain:N", axis=_axis(None, orient="top", labelAngle=0)),
        y=alt.Y("Category:N", sort=sorted(t["Category"].unique()), axis=_axis(None)),
    )
    cells = base.mark_rect(cornerRadius=4, stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color("average_pct:Q", scale=alt.Scale(domain=[-limit, 0, limit], range=[FALL, "#F2F3F1", RISE]),
                        legend=alt.Legend(title="Change (%)", orient="right")),
        tooltip=["Category", "Chain", alt.Tooltip("text:N", title="Average change"), alt.Tooltip("products:Q", title="Products")],
    )
    text = base.mark_text(fontSize=12).encode(
        text="text:N", color=alt.condition(f"abs(datum.average_pct) >= {0.6 * limit}", alt.value("#FFFFFF"), alt.value(INK)))
    return styled((cells + text).properties(height=_height(t["Category"].nunique(), 30)))

def matched_gap(table):
    """Histogram of price spread (gap_pct) across similarity-matched products."""
    valid = table["gap_pct"].dropna()
    if valid.empty:
        return styled(
            alt.Chart(pd.DataFrame({"message": ["No price gap data available"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=120)
        )

    width = 5
    capped = valid.clip(0, 100)
    centre = ((capped // width) * width + width / 2).astype(int)
    t = centre.value_counts().rename_axis("gap_mid").reset_index(name="products")
    t["range_label"] = t["gap_mid"].map(
        lambda v: f"{max(0, v - width // 2)}%–{v + width // 2}%"
    )
    t["tier"] = t["gap_mid"].map(
        lambda v: "Small (≤ 15%)" if v <= 15 else ("Moderate (15%–40%)" if v <= 40 else "Large (> 40%)")
    )

    bars = (
        alt.Chart(t)
        .mark_bar(cornerRadiusEnd=4, width=alt.RelativeBandSize(band=0.88))
        .encode(
            x=alt.X(
                "gap_mid:O",
                sort="ascending",
                title="Price gap between dearest and cheapest chain",
                axis=alt.Axis(
                    labelAngle=0,
                    labelExpr="datum.value + '%'",
                ),
            ),
            y=alt.Y("products:Q", title="Matched products"),
            color=alt.Color(
                "tier:N",
                scale=alt.Scale(
                    domain=["Small (≤ 15%)", "Moderate (15%–40%)", "Large (> 40%)"],
                    range=["#9CCDBB", "#1E5736", "#F0883E"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("range_label:N", title="Gap range"),
                alt.Tooltip("products:Q", title="Matched products"),
            ],
        )
    )

    return styled(bars.properties(height=250))

def basket_category_donut(basket_df, categories=None, bg_color="#F4EFE4"):
    """Donut chart showing basket spending share (%) and amount (€) by category."""
    def _donut_style(chart):
        return (
            chart.properties(
                autosize=alt.AutoSizeParams(type="fit-x", contains="padding"),
                padding={"left": 6, "right": 14, "top": 8, "bottom": 6},
            )
            .configure(font="Archivo", background=bg_color)
            .configure_view(fill=bg_color, strokeWidth=0)
            .configure_axis(
                labelColor=MUTED,
                titleColor=MUTED,
                gridColor="#E8EDEA",
                domainColor="#C9D2CD",
                labelFontSize=12,
                titleFontSize=12,
            )
            .configure_legend(
                labelColor=INK,
                titleColor=MUTED,
                labelFontSize=13,
                titleFontSize=13,
            )
        )

    if basket_df is None or not isinstance(basket_df, pd.DataFrame) or basket_df.empty:
        return _donut_style(
            alt.Chart(pd.DataFrame({"message": ["No basket items to display"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=180)
        )

    df = basket_df.loc[:, ~basket_df.columns.duplicated()].copy()

    cat_col = next((c for c in ["category", "Category", "cat"] if c in df.columns), None)
    amt_col = next(
        (c for c in ["amount_eur", "cost_eur", "price_eur", "total_eur", "subtotal_eur", "spend_eur"] if c in df.columns),
        None,
    )

    if amt_col is None:
        return _donut_style(
            alt.Chart(pd.DataFrame({"message": ["Missing spend column for category donut"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=180)
        )

    df["_amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

    if cat_col is not None:
        cat_series = df[cat_col].fillna("Uncategorized").astype(str)

        if isinstance(categories, dict) and categories:
            cat_series = cat_series.map(lambda c: str(categories.get(c, c)))
        elif callable(categories):
            cat_series = cat_series.map(lambda c: str(categories(c)))

        df["_category"] = cat_series
    else:
        df["_category"] = "Basket"

    grouped = (
        df.groupby("_category", as_index=False)["_amount"]
        .sum()
        .sort_values("_amount", ascending=False)
    )
    grouped = grouped[grouped["_amount"] > 1e-6].reset_index(drop=True)

    total_eur = float(grouped["_amount"].sum())
    if total_eur <= 0.0 or grouped.empty:
        return _donut_style(
            alt.Chart(pd.DataFrame({"message": ["Basket total is €0.00"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=180)
        )

    grouped["share_pct"] = grouped["_amount"] / total_eur * 100.0
    grouped["pct_label"] = grouped["share_pct"].map(lambda p: f"{p:.0f}%")
    grouped["legend_label"] = grouped.apply(
        lambda r: f"{r['_category']} ({r['share_pct']:.0f}% · €{r['_amount']:.2f})",
        axis=1,
    )

    def _lerp_hex(c1, c2, t):
        t = max(0.0, min(1.0, float(t)))
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(round(r1 + (r2 - r1) * t))
        g = int(round(g1 + (g2 - g1) * t))
        b = int(round(b1 + (b2 - b1) * t))
        return f"#{r:02X}{g:02X}{b:02X}"

    def _spaced_shades(n, start_hex, end_hex):
        if n <= 1:
            return [start_hex]
        raw = [_lerp_hex(start_hex, end_hex, i / (n - 1)) for i in range(n)]
        return raw[0::2] + raw[1::2]

    domain = grouped["legend_label"].tolist()
    n_cats = len(domain)
    n_green = (n_cats + 1) // 2
    n_orange = n_cats // 2

    green_shades = _spaced_shades(n_green, "#0F3D28", "#98D8BE")
    orange_shades = _spaced_shades(n_orange, "#C85519", "#F7B267")

    palette = []
    gi = oi = 0
    for idx in range(n_cats):
        if idx % 2 == 0 and gi < len(green_shades):
            palette.append(green_shades[gi])
            gi += 1
        elif oi < len(orange_shades):
            palette.append(orange_shades[oi])
            oi += 1
        elif gi < len(green_shades):
            palette.append(green_shades[gi])
            gi += 1

    base = alt.Chart(grouped).encode(
        theta=alt.Theta("_amount:Q", stack=True),
        color=alt.Color(
            "legend_label:N",
            scale=alt.Scale(domain=domain, range=palette[: len(domain)]),
            legend=alt.Legend(
                title="Category share",
                orient="right",
                offset=18,
                symbolSize=120,
                rowPadding=6,
                labelLimit=280,
                titleFontSize=13,
                labelFontSize=13,
            ),
        ),
        tooltip=[
            alt.Tooltip("_category:N", title="Category"),
            alt.Tooltip("_amount:Q", format=".2f", title="Spend (€)"),
            alt.Tooltip("share_pct:Q", format=".1f", title="Share (%)"),
        ],
    )

    arcs = base.mark_arc(
        innerRadius=54,
        outerRadius=92,
        stroke=bg_color,
        strokeWidth=2,
    )

    pct_labels = base.mark_text(radius=73, fontSize=11, fontWeight=700, color=SURFACE).encode(
        text=alt.condition("datum.share_pct >= 9", "pct_label:N", alt.value(""))
    )

    center_amount = (
        alt.Chart(pd.DataFrame({"txt": [f"€{total_eur:.2f}"]}))
        .mark_text(dy=-6, fontSize=15, fontWeight=700, color=INK)
        .encode(text="txt:N")
    )

    center_sub = (
        alt.Chart(pd.DataFrame({"txt": ["Plan items"]}))
        .mark_text(dy=11, fontSize=10.5, color=MUTED)
        .encode(text="txt:N")
    )

    chart_height = max(225, min(420, len(grouped) * 26))
    return _donut_style((arcs + pct_labels + center_amount + center_sub).properties(height=chart_height))


# ============================================================
# SHOPPER VIEW AND NEW ANALYST CHARTS (24 Nov)
# ============================================================

def list_cost(table, colors, names):
    """What the same shopping list costs at each chain, buying the cheapest product of each type."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)))
    t["text"] = [f"{c:.2f} €" + ("  cheapest" if e < 0.005 else f"  +{e:.2f} €")
                 for c, e in zip(t["cheapest"], t["extra"])]
    order = t["Chain"].tolist()
    y = alt.Y("Chain:N", sort=order, axis=_axis(None, labelFontSize=13, labelColor=INK))
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("cheapest:Q", title="List cost (€)", axis=alt.Axis(tickCount=5),
                scale=alt.Scale(domain=[0, float(t["cheapest"].max()) * 1.3])),
        y=y, color=alt.Color("Chain:N", scale=chain_scale(colors, names), legend=None),
        tooltip=["Chain", alt.Tooltip("cheapest:Q", format=".2f", title="Cheapest products (€)"),
                 alt.Tooltip("typical:Q", format=".2f", title="Typical products (€)"),
                 alt.Tooltip("extra:Q", format="+.2f", title="Over the cheapest chain (€)")],
    )
    text = alt.Chart(t).mark_text(align="left", dx=6, fontSize=12, color=INK).encode(x="cheapest:Q", y=y, text="text:N")
    return styled((bars + text).properties(height=_height(len(order), 38)))


def chain_premium(table, colors, names, min_products=100):
    """On identical (engine-matched) products: share of them where each chain is the cheapest.

    Chains with fewer than min_products matched products are faded: their share rests on little data."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)))
    t["text"] = [f"{c:.0f}%  ·  avg +{p:.0f}% over cheapest  ·  {n} products" + ("  (few)" if n < min_products else "")
                 for c, p, n in zip(t["cheapest_pct"], t["premium_pct"], t["products"])]
    order = t.sort_values("cheapest_pct", ascending=False)["Chain"].tolist()
    y = alt.Y("Chain:N", sort=order, axis=_axis(None))
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("cheapest_pct:Q", title="Share of its matched products where the chain is the cheapest (%)",
                scale=alt.Scale(domain=[0, 140]), axis=alt.Axis(values=[0, 25, 50, 75, 100])),
        y=y, color=alt.Color("Chain:N", scale=chain_scale(colors, names), legend=None),
        opacity=alt.condition(f"datum.products < {min_products}", alt.value(0.4), alt.value(1)),
        tooltip=["Chain", alt.Tooltip("cheapest_pct:Q", format=".0f", title="Cheapest (%)"),
                 alt.Tooltip("premium_pct:Q", format=".1f", title="Average % over cheapest"),
                 alt.Tooltip("products:Q", title="Matched products")],
    )
    text = alt.Chart(t).mark_text(align="left", dx=6, fontSize=11.5, color=INK).encode(x="cheapest_pct:Q", y=y, text="text:N")
    return styled((bars + text).properties(height=_height(len(order), 34)))


def own_brand_map(table, colors, names):
    """Own brand: how much of the shelf it is (x) against how much it saves (y). One dot per chain."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x))).reset_index(drop=True)
    # Labels go right of the dot; if that box would hit another dot or label, try left, above, below.
    # Boxes in data units (x 0–70 ≈ 900 px, y 0–60 ≈ 380 px).
    placed, sides = [], []
    offsets = {"right": (1.0, 0), "left": (-1.0, 0), "above": (0, 3.2), "below": (0, -3.2)}
    for r in t.itertuples():
        w = 0.62 * len(r.Chain)
        for side, (dx, dy) in offsets.items():
            x0 = r.own_share_pct + (dx if side != "left" else dx - w) - (w / 2 if dx == 0 else 0)
            box = (x0, x0 + w, r.gap_pct + dy - 1.3, r.gap_pct + dy + 1.3)
            hits_dot = any(box[0] - 0.8 <= o.own_share_pct <= box[1] + 0.8 and box[2] - 1 <= o.gap_pct <= box[3] + 1
                           for o in t.itertuples() if o.Index != r.Index)
            hits_label = any(box[0] < b[1] and b[0] < box[1] and box[2] < b[3] and b[2] < box[3] for b in placed)
            if not (hits_dot or hits_label):
                break
        placed.append(box)
        sides.append(side)
    t["side"] = sides
    base = alt.Chart(t).encode(
        x=alt.X("own_share_pct:Q", title="Own brand share of the shelf (%)", scale=alt.Scale(domain=[0, 70])),
        y=alt.Y("gap_pct:Q", title="Saving against national brands (%)", scale=alt.Scale(domain=[0, 60])),
        tooltip=["Chain", alt.Tooltip("own_share_pct:Q", format=".0f", title="Own brand share (%)"),
                 alt.Tooltip("gap_pct:Q", format=".0f", title="Saving (%)"), alt.Tooltip("types:Q", title="Types compared")],
    )
    dots = base.mark_circle(size=220, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color("Chain:N", scale=chain_scale(colors, names), legend=None))
    layers = [dots]
    for side, mark in {"right": dict(align="left", dx=12), "left": dict(align="right", dx=-12),
                       "above": dict(align="center", dy=-16), "below": dict(align="center", dy=16)}.items():
        if (t["side"] == side).any():
            layers.append(alt.Chart(t[t["side"] == side]).mark_text(fontSize=12, fontWeight=600, color=INK, **mark).encode(
                x="own_share_pct:Q", y="gap_pct:Q", text="Chain:N"))
    return styled(alt.layer(*layers).properties(height=320))


def optimizer_savings(table):
    """Saving of the exact optimizer over the price list, one column per random basket (largest first)."""
    t = table.sort_values("saving_eur", ascending=False).reset_index(drop=True)
    t["rank"] = range(1, len(t) + 1)
    t["Result"] = t["saving_eur"].map(lambda v: "Cheaper with the optimizer" if v > 0.005 else "Same cost")
    t["plot"] = t["saving_eur"].clip(lower=0.05)  # a stub so "same cost" baskets stay visible
    domain = ["Cheaper with the optimizer", "Same cost"]
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=3, width=alt.RelativeBandSize(0.8)).encode(
        x=alt.X("rank:O", title="Random baskets, largest saving first", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("plot:Q", title="Saving (€)"),
        color=alt.Color("Result:N", scale=alt.Scale(domain=domain, range=[BEST, QUIET]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("saving_eur:Q", format=".2f", title="Saving (€)"),
                 alt.Tooltip("greedy_eur:Q", format=".2f", title="Price list total (€)"),
                 alt.Tooltip("exact_eur:Q", format=".2f", title="Optimizer total (€)"),
                 alt.Tooltip("greedy_stops:Q", title="Price list stops"), alt.Tooltip("exact_stops:Q", title="Optimizer stops")],
    )
    return styled(bars.properties(height=240))


def price_over_time(series, colors, names, unit_text):
    """One line per chain over the collection dates; every chain that sells the product is in the legend."""
    t = series.assign(Chain=series["supermarket"].map(lambda x: names.get(x, x)),
                      date=pd.to_datetime(series["collection_date"]))
    present = [k for k in colors if k in set(series["supermarket"])]
    scale = alt.Scale(domain=[names.get(k, k) for k in present], range=[colors[k] for k in present])
    tips = [alt.Tooltip("Chain:N"), alt.Tooltip("date:T", format="%d %b %Y", title="Date"),
            alt.Tooltip("price:Q", format=".2f", title=f"€/{unit_text}")]
    if "pack_price" in t:
        tips.append(alt.Tooltip("pack_price:Q", format=".2f", title="Pack price (€)"))
    if "products" in t:
        tips.append(alt.Tooltip("products:Q", title="Products in the median"))
    hover = alt.selection_point(fields=["Chain"], on="pointerover", nearest=True, empty=False)
    # The axis spans at least ±10% of the middle price, so a 5-cent move doesn't look like a crash
    lo, hi = float(t["price"].min()), float(t["price"].max())
    mid, half = (lo + hi) / 2, max((hi - lo) / 2 * 1.15, 0.1 * (lo + hi) / 2)
    base = alt.Chart(t).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, grid=False,
                                                     tickCount={"interval": "week", "step": 1}, labelOverlap=True)),
        y=alt.Y("price:Q", title=f"€ per {unit_text}", scale=alt.Scale(domain=[max(0, mid - half), mid + half], nice=True)),
        color=alt.Color("Chain:N", scale=scale,
                        legend=alt.Legend(orient="top", title=None, symbolType="stroke", symbolStrokeWidth=3)),
    )
    lines = base.mark_line().encode(strokeWidth=alt.condition(hover, alt.value(3.5), alt.value(2)))
    points = base.mark_circle(size=80, opacity=1, stroke=SURFACE, strokeWidth=2).encode(tooltip=tips)
    hits = base.mark_circle(size=500, opacity=0).encode(tooltip=tips).add_params(hover)  # bigger hover target
    return styled((lines + points + hits).properties(height=340))
