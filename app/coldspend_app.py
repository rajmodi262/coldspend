"""Coldspend — the interactive decision app.

Runs as a normal marimo notebook locally, and exports to WebAssembly for
GitHub Pages, where the whole thing executes in the viewer's browser via
Pyodide. There is no application server and no database — nothing to pay for and
nothing that can fall over mid-demo.

ONE HONEST QUALIFICATION, because "no server" is easy to overclaim: marimo does
NOT bundle Pyodide. The runtime and the scientific stack (~30 MB) are fetched at
load time from cdn.jsdelivr.net and then cached by the browser. So the page has
one external dependency and a slow first paint. What is true is that no
application code runs remotely and no data leaves the machine.

    marimo edit app/coldspend_app.py                       # develop
    marimo export html-wasm app/coldspend_app.py \\
        -o site/app --mode run                             # ship

The optimiser really does solve client-side. `scipy.optimize.milp` IS HiGHS and
scipy ships with Pyodide, which is the whole reason this architecture is
possible — `highspy` has no emscripten wheel.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Coldspend")


@app.cell
async def _():
    # In WebAssembly there is no site-packages, so the package is installed from
    # a wheel served next to this page. Locally it is already importable and this
    # falls straight through.
    #
    # The URL MUST be resolved with mo.notebook_location(). A relative path like
    # "./coldspend-...whl" resolves against the WEB WORKER's location, not the
    # page's, so it silently fetches a 404 HTML page and micropip fails with
    # `BadZipFile: File is not a zip file` — which is what the first version of
    # this app did, and why nothing below it ever ran.
    import marimo as mo

    # Imported here so marimo's dependency scanner preloads them into Pyodide.
    # It can only see imports written in the notebook itself — scipy is imported
    # inside coldspend.decide.optimizer, which the scanner never reads, so
    # leaving it implicit produced `ModuleNotFoundError: No module named 'scipy'`
    # and killed every cell below.
    # The Agg backend must be selected BEFORE pyplot is imported. Under Pyodide
    # there is no display, and pyplot picking an interactive backend is a known
    # way for figures to silently produce no output.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy  # noqa: F401

    try:
        import coldspend  # noqa: F401
    except ImportError:
        import micropip

        # Install the runtime dependencies EXPLICITLY, then the package with
        # deps=False. The wheel declares the whole project's requirements —
        # pandas, pyarrow, lifelines, diskcache — and several have no emscripten
        # build, so letting micropip resolve them fails. But deps=False alone is
        # not enough either: it skips scipy, which this app genuinely needs.
        await micropip.install(["numpy", "scipy", "matplotlib"])
        wheel = str(mo.notebook_location() / "public" / "coldspend-0.1.0-py3-none-any.whl")
        await micropip.install(wheel, deps=False)
    from coldspend.decide import (
        DEFAULT_COSTS,
        Action,
        ShipmentState,
        decision_stability,
        policy_map,
        recommend,
    )
    from coldspend.decide.optimizer import ACTIONS, project_budget
    from coldspend.physics import equivalent_hours
    return (
        ACTIONS, Action, DEFAULT_COSTS, ShipmentState, decision_stability,
        equivalent_hours, mo, np, plt, policy_map, project_budget, recommend,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Coldspend

        ### Cold chain is not a limit you breach. It's a budget you spend.

        Two shipments. Both 60 hours, **both entirely inside 2–8 °C**, both filing zero
        deviations, both passing compliance identically. One cruises at 3.0 °C, the other at
        7.5 °C. The warm one ages the product **1.787× faster**.

        Compliance cannot tell them apart. A stability budget can — and once you have a
        budget instead of an alarm, the question stops being *"did we breach?"* and becomes
        **"what is the cheapest thing to do about it right now?"**

        Everything below is computed in your browser. There is no application server and no
        database — the optimiser genuinely solves locally, on your machine. To be precise
        about it: the Python runtime itself (Pyodide, plus numpy and scipy) is fetched once
        from a public CDN and then cached, so first load is slow and everything after it is
        instant. Nothing is *sent* anywhere.
        """
    )
    return


@app.cell
def _(mo):
    value = mo.ui.dropdown(
        options={
            "Saline — $4,000": 4_000.0,
            "Insulin shipment — $45,000": 45_000.0,
            "Vaccine pallet — $120,000": 120_000.0,
            "mAb pallet — $850,000": 850_000.0,
            "CAR-T dose — $443,600": 443_600.0,
        },
        value="mAb pallet — $850,000",
        label="Consignment",
    )
    tau = mo.ui.slider(
        4, 72, value=12, step=2, label="Packaging quality — thermal time constant τ (hours)",
        show_value=True,
    )
    cost_t = mo.ui.slider(
        0.0, 1.0, value=0.5, step=0.25,
        label="Where in the cost range are we? (0 = low, 1 = high)", show_value=True,
    )
    mo.vstack([mo.md("## The policy map"), value, tau, cost_t])
    return cost_t, tau, value


@app.cell
def _(Action, ACTIONS, cost_t, mo, np, plt, policy_map, tau, value):
    # 56x56, not 120x120. The map is recomputed on every slider drag, and each
    # cell is an argmin over the action space — 120x120 is 14,400 optimiser calls
    # per redraw, which is fine natively and unusably slow under Pyodide. At 56
    # the decision boundaries are still smooth to the eye and the map redraws in
    # about a second.
    pm = policy_map(value.value, tau_h=float(tau.value), n_budget=56, n_hours=56,
                    t=float(cost_t.value))

    colour = {
        Action.DO_NOTHING: "#e6e5df", Action.REICE: "#2a78d6",
        Action.EXPEDITE: "#eb6834", Action.REROUTE: "#1baf7a",
        Action.RECALL: "#e34948",
    }
    present = sorted(np.unique(pm.action_index))
    cmap = plt.matplotlib.colors.ListedColormap([colour[ACTIONS[int(i)]] for i in present])
    remap = np.zeros_like(pm.action_index)
    for new, old in enumerate(present):
        remap[pm.action_index == old] = new

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.pcolormesh(pm.budget_grid, pm.hours_grid, remap, cmap=cmap, shading="auto",
                  vmin=-0.5, vmax=len(present) - 0.5)
    for new, old in enumerate(present):
        act = ACTIONS[int(old)]
        ys, xs = np.where(remap == new)
        if ys.size < 40:
            continue
        ax.annotate(
            act.value.replace("_", " "),
            (pm.budget_grid[int(np.median(xs))], pm.hours_grid[int(np.median(ys))]),
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="white" if act is not Action.DO_NOTHING else "#52514e",
        )
    ax.set_xlabel("stability budget already consumed (%)")
    ax.set_ylabel("hours remaining to destination")
    ax.set_title("Cheapest action from every state", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()

    shares = "  ·  ".join(
        f"**{a.value.replace('_', ' ')}** {s:.0%}"
        for a, s in sorted(pm.region_shares().items(), key=lambda kv: -kv[1])
    )
    mo.vstack([fig, mo.md(f"Region shares: {shares}")])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Re-icing buys time, not immunity.** A fresh coolant charge holds the payload down for
        roughly one thermal time constant; after that the box is driven by ambient again. Drag
        τ upward and watch *expedite* give way to *re-ice* — with the same product, the same
        money and the same risk. That boundary is the whole argument for pricing actions
        against each other instead of following a fixed playbook.

        A map like this can be reviewed and signed **once, in advance**, by a Qualified Person.
        That is a very different governance object from a recommendation somebody has to
        adjudicate shipment by shipment at 2 a.m.
        """
    )
    return


@app.cell
def _(mo):
    budget = mo.ui.slider(0, 95, value=34, step=1, label="Budget already consumed (%)",
                          show_value=True)
    hours = mo.ui.slider(2, 60, value=22, step=1, label="Hours remaining", show_value=True)
    burn = mo.ui.slider(0.2, 3.0, value=1.4, step=0.1, label="Current burn rate (%/h)",
                        show_value=True)
    p_exc = mo.ui.slider(0.0, 1.0, value=0.62, step=0.02, label="P(excursion) if nothing is done",
                         show_value=True)
    mo.vstack([mo.md("## One shipment, right now"), budget, hours, burn, p_exc])
    return budget, burn, hours, p_exc


@app.cell
def _(Action, ShipmentState, budget, burn, cost_t, hours, mo, p_exc,
      project_budget, recommend, tau, value):
    s = ShipmentState(
        shipment_id="LIVE", hub="DXB",
        budget_pct=float(budget.value), burn_rate_pct_per_h=float(burn.value),
        hours_remaining=float(hours.value), consignment_value_usd=value.value,
        p_excursion=float(p_exc.value), tau_h=float(tau.value),
    )
    best, losses = recommend(s, t=float(cost_t.value))
    do_nothing = losses[Action.DO_NOTHING]
    benefit = do_nothing - losses[best]

    rows = "\n".join(
        f"| {'**' if a is best else ''}{a.value.replace('_', ' ')}"
        f"{'**' if a is best else ''} | ${v:,.0f} | "
        f"{'—' if a is Action.DO_NOTHING else f'${do_nothing - v:+,.0f}'} |"
        for a, v in sorted(losses.items(), key=lambda kv: kv[1])
    )

    verdict = (
        f"### Recommendation: **{best.value.replace('_', ' ')}**\n\n"
        f"Projected budget at arrival if nothing is done: "
        f"**{project_budget(s, Action.DO_NOTHING):.0f}%** of label. "
        + (f"Taking this action avoids **${benefit:,.0f}** of expected loss."
           if benefit > 0 else "No action beats doing nothing here.")
    )

    mo.vstack([
        mo.md(verdict),
        mo.md(f"| action | expected total cost | vs doing nothing |\n|---|---|---|\n{rows}"),
        mo.callout(
            mo.md(
                "Expected cost = intervention cost + P(spoil) × consignment value + "
                "P(excursion) × QA deviation investigation. That **third term is why doing "
                "nothing is rarely free** — an excursion costs an investigation whether or "
                "not the product is ultimately released. On a $4,000 saline shipment the "
                "investigation costs several times the goods, so intervening protects the "
                "paperwork rather than the product."
            ),
            kind="info",
        ),
    ])
    return


@app.cell
def _(decision_stability, mo, value):
    rep = decision_stability(value.value, n_budget=22, n_hours=22)
    mo.callout(
        mo.md(
            f"## Does this survive the costs being wrong?\n\n"
            f"**{rep.stable_share:.0%} of the state space keeps the same recommendation** "
            f"across the full low-to-high cost range ({rep.n_stable:,} of {rep.n_states:,} "
            f"states).\n\nThat is deliberately a stronger claim than any dollar figure. The "
            f"intervention costs here are assumptions in a plausible band — a reviewer can "
            f"reject every one of them individually and the recommended *action* still "
            f"stands. Decision stability beats point precision."
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ### What this is, and what it isn't

        **Simulated.** No proprietary telemetry was used. Shipments come from a physics-based
        digital twin — a lumped-capacitance RC thermal model with phase-change handling, Mean
        Kinetic Temperature per USP ⟨1079⟩, and Arrhenius degradation integrated over the
        trace.

        **Prediction is not the contribution.** Excursion prediction here scores ~0.98 AUC, and
        that is a warning rather than a trophy: forecasting a breach from *"already at 7.8 °C
        and warming"* is genuinely easy. It is also already patented (US 11,769,103 B2) and
        already shipping. What is missing everywhere is pricing every mid-transit action in one
        currency against a continuous stability budget and taking the argmin.

        **Known limitation.** The regression-discontinuity estimate elsewhere in this project
        covers the truth and is stable across bandwidths, but still overstates the effect by
        roughly 50%; robust bias-correction is not yet applied. The policy map does not depend
        on it.

        [Source and full write-up →](https://github.com/rajmodi262/coldspend)
        """
    )
    return


if __name__ == "__main__":
    app.run()
