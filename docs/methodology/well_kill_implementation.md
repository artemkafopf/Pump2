# Well Kill Implementation — Radial Multiphase Flow Simulator

## Context: TGT-H1-5P Completion

The reference well driving this design is a deviated gas-lifted oil producer with the following
characteristics relevant to kill modeling:

| Item | Value |
|---|---|
| Max inclination | 46.6° at 2000 m MDRT |
| TD | 3533 m MDRT / ~2476 m TVDRT |
| Upper tubing | 5-1/2", 23# (ID 4.670"), 0–124 m |
| Lower tubing | 4-1/2", 15.1# (ID 3.826"), 124–2885 m |
| Gas lift | Side Pocket Mandrel at ~1121 m MDRT |
| Perforations | 2927–3219 m MDTH across zones ULBH, 5.1, 5.2U, 5.2L |
| Pack-offs | ~3105 m and ~3125.5 m MDTH (isolating lower 5.2U intervals) |
| Completion fluid | 8.8 ppg NaCl |

The pack-offs divide the perforated section into hydraulically semi-isolated compartments.
Different zones were perforated at different times (2011, 2016, 2024) and are at different
depletion states.

---

## Kill Procedure Choice

### Why Bullhead

Bullhead is the appropriate procedure for this well because:

- No workover string or coiled tubing on bottom — circulation kill is impossible
- Multiple perforation intervals spread over ~300 m — no clean circulatable path
- Gas-lifted producer — gas supply can be shut in at surface, after which the well
  is killed by pumping heavy fluid down the tubing to overpower reservoir influx

Displacement pumps at fixed surface rate are standard for bullhead operations. The simulator
uses **rate-controlled injection** throughout the kill sequence.

### Kill Feasibility (Static Check)

Kill is feasible only if the static hydrostatic of the kill fluid column exceeds the
maximum reservoir pressure across all open zones:

```
ρ_kill × g × TVD_deepest_perf  ≥  P_res_max + P_surface_backpressure
```

Key points:

- **TVD, not MD.** For a well deviated to 47°, TVD is ~24% shorter than MD.
  At 3200 m MD to the deepest perf, TVD is ~2476 m — a significant penalty vs vertical.
- **Pump pressure limit.** If the kill fluid hydrostatic alone is insufficient, the pump
  contributes `P_surface_pump`. The kill is infeasible if
  `ρ_kill × g × TVD + P_pump_max < P_res_max`.
- **Kill fluid density.** Must be chosen so the static column exceeds `P_res_max`.
  Standard options: NaCl brine (max ~10 ppg), CaCl₂ (~11.6 ppg), CaBr₂/ZnBr₂ (~>14 ppg).
- **Depth alone does not guarantee kill.** Overpressured reservoirs or gas-dominated
  wellbores (low effective hydrostatic) can defeat a bullhead regardless of depth.

A `kill_feasibility_check()` method should evaluate this statically before the simulation
begins.

---

## Correct Kill Criterion

### The Common Confusion

During rate-controlled injection, BHP at every perforation exceeds the local cell pressure
by definition — that is what drives flow into the reservoir. The instantaneous cell pressure
`P_cell` during active injection is **not** the kill criterion.

### The Correct Criterion

Kill is established when the **static** hydrostatic column of kill fluid alone — pump
contribution zero — can hold the well against reservoir influx:

```
Kill established ⟺  ρ_kill × g × TVD_i  ≥  P_res_i     for all open perforation intervals i
```

where `P_res_i` is the **far-field reservoir pressure** of zone `i` — the pressure that
zone would push at if the well were shut in and allowed to fully build up in isolation.

In the simulator, this is evaluated as:

```python
kill_established = all(
    rho_kill * g * tvd(perf) >= reservoir_cell[perf].pressure + margin
    for perf in open_perforations
)
```

using live cell pressures each timestep. The kill criterion must be satisfied against the
**highest-pressure zone**, because that zone determines whether the well flows back after
the pump stops.

---

## Inter-Layer Crossflow During Shut-In

### Mechanism

When the well is shut in with zones at different pressures, the wellbore becomes a conduit
between them. A high-pressure zone (aquifer-connected) flows into the wellbore; the wellbore
pressure pushes into the low-pressure zone (depleted reservoir). This is wellbore crossflow.

The shut-in wellbore pressure (SIWHP) settles at the injectivity-weighted average:

```
P_wf_shutin  =  Σᵢ (II_i × P_res_i) / Σᵢ II_i
```

This is **not** equal to the reservoir pressure of any individual zone.

### Finite vs Persistent Crossflow

| Scenario | Crossflow duration | SIWHP behavior |
|---|---|---|
| All zones closed (no boundary support) | Finite — decays as pressures equalize | Plateaus at II-weighted average |
| One zone has constant-pressure boundary (aquifer) | **Indefinite** | Plateaus near P_aquifer (aquifer dominates) |

In the aquifer case, waiting longer during shut-in makes the kill **harder**: the aquifer
continuously recharges the high-pressure zone, which crossflows into the depleted zone through
the wellbore, pressurizing it toward P_aquifer. The effective `P_res_max` rises over time.

### Implications for the Kill Procedure

- Do not use SIWHP as a single `P_res_reference` — it is an II-weighted mix.
- The simulator has per-zone `P_cell_i` directly; use that for the kill criterion.
- Begin pumping before prolonged crossflow pressurizes depleted zones unnecessarily.
- In the aquifer scenario, the aquifer-connected zone defines the pressure ceiling the
  kill must overcome. This zone's `P_cell` must be checked in the kill criterion even if
  it has not accepted significant kill fluid.

### Boundary Condition Effects on Kill

| Outer boundary type | P_res behavior during injection | Kill difficulty |
|---|---|---|
| Constant pressure (open aquifer) | Stays near P_boundary despite injection | Hardest — must overcome fixed P_boundary |
| Impermeable (closed) | Rises with every barrel injected | Gets easier over time; less fluid loss |
| Free-flow (depleted, open) | May be below initial; essentially fixed | Easiest — low P_res to overcome |

---

## Required Model Changes

### 1. WellboreColumn — 1D Front Tracking

The current average-density BHP is insufficient for kill simulation. The wellbore contains a
sharp compositional front: kill fluid above, original wellbore fluid below. The BHP at any
depth depends on the actual density profile, not the average:

```
BHP(z_perf) = P_surface + ∫₀^z_perf  ρ(z) · g · cos θ(z)  dz

where  ρ(z) = ρ_kill    if  z < z_front
              ρ_wb       if  z > z_front
```

Using the average density underestimates BHP during mid-kill (front at intermediate depth)
and is especially wrong when gas is present in the lower wellbore section below the front.

#### Data Structure

```python
@dataclass
class WellboreSegment:
    depth_top: float        # m MDRT
    depth_bot: float
    rho: float              # kg/m³, current fluid density in this segment
    cross_section: float    # m², varies (5.5" above 124m, 4.5" below)

class WellboreColumn:
    segments: list[WellboreSegment]
    z_front: float          # MD position of kill fluid front (m)
    rho_kill: float         # kg/m³
    rho_wb: float           # kg/m³, original wellbore fluid

    def advance_front(self, rate: float, dt: float):
        dz = rate * dt / self.cross_section_at(self.z_front)
        self.z_front += dz

    def hydrostatic_bhp(self, P_surface: float, deviation_profile) -> float:
        bhp = P_surface
        for seg in self.segments:
            rho = self.rho_kill if seg.depth_top < self.z_front else self.rho_wb
            bhp += rho * G * seg.dz * cos(deviation_profile.theta(seg.depth_top))
        return bhp
```

#### Wellbore Storage

The wellbore volume is large (~13–15 m³ for ~2885 m of 4.5" tubing). It buffers pressure
changes during GL shut-in and early kill. Add:

```python
class Well:
    C_wb: float   # m³/Pa — wellbore storage coefficient
    # Liquid-filled: C_wb ≈ V_wb × c_fluid  ≈ 13 m³ × 4e-10 Pa⁻¹ ≈ 5e-9 m³/Pa
    # Gas-dominated: C_wb ≈ V_wb / P_wb     (much larger, sluggish response)
```

Set `C_wb` based on the wellbore fluid state at the time of shut-in. A gas-dominated wellbore
has much larger WBS and takes much longer to respond to pumping.

### 2. Kill Fluid Properties

```python
@dataclass
class KillFluidProps:
    density: float          # kg/m³
    viscosity: float        # Pa·s
    compressibility: float  # Pa⁻¹

class Well:
    kill_fluid: KillFluidProps | None = None
    wellbore_column: WellboreColumn | None = None
```

### 3. Injected Fluid Composition Per Perforation

As the kill front descends, different perforation intervals receive different fluids:

- Before the front reaches a perf: the original wellbore fluid (oil/gas/brine mix) is
  what enters the reservoir at that interval — the kill front is pushing wellbore fluid ahead.
- After the front passes a perf: kill brine is entering the reservoir at that interval.

Upper perfs (shallower) receive kill brine first. Lower perfs receive wellbore fluid for
longer. The pack-off-isolated lower intervals receive wellbore fluid until the front descends
through the full tubing length.

```python
def injected_fluid_at_perf(self, perf) -> FluidComposition:
    if self.wellbore_column.z_front >= perf.depth_md:
        return self.kill_fluid
    else:
        return self.original_wellbore_fluid
```

This composition is passed to the existing per-zone injectivity allocation, which handles
the rate split. The reservoir cells then track saturation of kill brine vs formation fluid
in the usual way.

### 4. Gas Lift Handling

The SPM at ~1121 m MDRT contains a wireline-retrievable gas lift valve linking the
tubing-casing annulus to the tubing. During normal production, gas is injected through the
annulus and enters the tubing at 1121 m, aerating the fluid column.

For kill simulation:
- Close the GL connection at 1121 m before starting the kill (shut in GL supply at surface).
- In the model: set transmissibility of the annulus-tubing link at this node to zero.
- The annular volume above 1121 m contributes to wellbore storage but does not participate
  in flow during the kill.
- No other model changes are required for GL. The wellbore is treated as a single-string
  system (tubing only) during kill mode.

### 5. WellManager — Kill Mode State Machine

```python
class WellMode(Enum):
    PRODUCTION_BHP   = "bhp_prod"
    PRODUCTION_RATE  = "rate_prod"
    INJECTION_RATE   = "rate_inj"
    INJECTION_BHP    = "bhp_inj"
    BULLHEAD_KILL    = "bullhead_kill"
    SHUT_IN          = "shut_in"

@dataclass
class BullheadKillSpec:
    surface_pump_rate: float        # m³/s
    surface_pump_pressure_max: float # Pa, pump limit
    kill_fluid: KillFluidProps
    kill_margin: float              # Pa above P_res to declare kill (e.g. 0.5 MPa)
    equilibration_rate_threshold: float  # m³/s below which Phase 6 is considered complete
```

---

## Step-by-Step Simulator Procedure

### Phase 0 — BHP-Controlled Production (starting state)

```
Well mode:      BHP_PRODUCTION
BHP:            P_prod  (< P_res, well flowing)
Wellbore fluid: original production mixture (oil/gas/brine)
WellboreColumn: not initialized
```

### Phase 1 — Shut In

```
Well mode:  SHUT_IN  →  Q = 0
```

1. Set gas lift connection transmissibility at 1121 m to zero.
2. Initialize `WellboreColumn` filled entirely with current wellbore fluid; `z_front = 0`.
3. Run timesteps. Wellbore pressure rises as WBS dissipates on timescale `C_wb / II_total`.
4. **Record `P_res_reference_i` per zone** — the live `P_cell_i` for each perforation interval
   at the end of the buildup period. Do not use SIWHP as a single reference.
5. If inter-zone crossflow is present (aquifer + depleted zone), note that the depleted zone
   pressure is rising. Begin kill before it equalizes fully to the aquifer pressure.

### Phase 2 — Rate-Controlled Kill (BULLHEAD_KILL)

```
Well mode:  BULLHEAD_KILL
Q:          Q_pump  (fixed)
```

Each timestep:

**a. Advance kill front**

```python
z_front += Q_pump * dt / wellbore_column.cross_section_at(z_front)
```

**b. Compute BHP from density profile integral**

```python
BHP = wellbore_column.hydrostatic_bhp(P_surface=0, deviation_profile)
# P_surface solved from: BHP = P_surface + hydrostatic → P_surface = BHP - hydrostatic
# In rate-controlled mode: Q is fixed, BHP is computed, P_surface is the observable
```

**c. Assign injected fluid per perforation**

```python
for perf in open_perforations:
    perf.injected_fluid = kill_brine if z_front >= perf.depth_md else original_wb_fluid
```

**d. Per-zone rate allocation**

Existing injectivity allocation handles this. Each zone receives:

```
q_i  =  II_i × (BHP_at_perf_i - P_cell_i)
```

Zones with lower `P_cell_i` accept more fluid initially (more depleted zones fill faster).

**e. Log surface pump pressure**

```python
P_surface = BHP - wellbore_column.hydrostatic_bhp(P_surface=0) + friction_losses
```

This should decrease monotonically as kill fluid fills the wellbore (heavier column does more
work, pump does less for the same BHP). A rising pump pressure indicates unexpected reservoir
pressure or gas influx.

**f. Check kill criterion**

```python
kill_established = all(
    rho_kill * G * tvd(perf) >= reservoir_cell[perf].pressure + kill_margin
    for perf in open_perforations
)
```

Note: `reservoir_cell[perf].pressure` is the live cell pressure, which may have risen from
injection. The kill criterion automatically becomes easier to satisfy in closed-boundary zones
(rising P_cell closes the gap on the static hydrostatic side) and harder in constant-pressure
boundary zones (P_cell stays fixed at P_boundary).

### Phase 3 — Equilibration

Once `kill_established = True`:

```
Well mode:  INJECTION_BHP
BHP:        rho_kill × g × TVD_deepest_perf   (static kill hydrostatic, pump off)
```

Continue running the simulation. Observe per-zone injection rate at this fixed BHP:

| Observed rate | Meaning |
|---|---|
| Small positive `q_i` for some zones | Overbalance pushing kill fluid slowly into zone — additional fluid loss accumulating |
| All `q_i → 0` | Equilibrated, kill is stable and complete |
| Any `q_i < 0` | Reservoir flowing back — kill failed at that zone, return to Phase 2 |

Total kill fluid loss = cumulative injection from Phase 2 + Phase 3.

---

## Kill Fluid Loss Components

The simulator computes three additive components:

| Component | Description | Computed by |
|---|---|---|
| Wellbore displacement | Volume of fluid pushed ahead of kill front before it reaches perfs (~21 m³ for this well) | `z_front` integral — this is wellbore fluid, not kill brine |
| Dynamic loss (Phase 2) | Kill brine injected into reservoir while BHP is climbing to kill condition | Cumulative `q_i` after kill brine reaches perfs |
| Equilibration loss (Phase 3) | Seepage under static overbalance | Cumulative `q_i` in BHP-control mode |

The wellbore displacement volume (~21 m³) is not strictly "lost" kill brine — it is the
original wellbore fluid entering the reservoir. Whether this is problematic depends on
the operation (e.g., if kill brine incompatibility with formation is a concern).

---

## Surface Pump Pressure as Kill Progress Indicator

During Phase 2, `P_surface` is the primary observable. Its trajectory reveals the kill state:

```
Early kill:   P_surface high   — light wellbore fluid in tubing, pump fighting reservoir alone
Mid-kill:     P_surface falling — kill fluid column growing, hydrostatic contributing more
Late kill:    P_surface low     — wellbore nearly full of kill brine
Kill secured: P_surface → 0     — hydrostatic alone exceeds P_res, pump no longer needed
```

If `P_surface` reaches zero while still pumping at `Q_pump`, the pump can be stopped and the
well is killed by statics alone. This is the field-observable confirmation of the kill
criterion.

---

## What Is Not Changed

- Reservoir solve (pressure, saturation equations) — unchanged
- Per-zone injectivity allocation — already implemented, used as-is
- Saturation tracking in reservoir cells — handles the kill brine / formation fluid
  displacement automatically once injected fluid composition is set correctly per perf
- Outer boundary well (constant P, rate, or impermeable) — unchanged

The production code path (BHP or rate controlled production) is unaffected. The
`WellboreColumn` is only activated when `mode == BULLHEAD_KILL`.
