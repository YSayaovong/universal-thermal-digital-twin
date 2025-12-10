"""
Universal Thermal Digital Twin - Senior Version (v3)

Features:
- Lumped-parameter thermal model with multiple nodes
- Convection to ambient
- Conduction between nodes
- Time-step simulation (explicit Euler)
- Hot-spot detection and thermal margin vs failure threshold
- Parametric sensitivity analysis:
    * Load vs peak temperature
    * Cooling efficiency (hA) vs peak temperature
    * Ambient temperature vs peak temperature
- Overload-duration study for thermal risk assessment
- Plots for all key studies (if matplotlib is installed)

Author: Yengkong V. Sayaovong
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Core Model Types
# ---------------------------------------------------------------------------

HeatGenFn = Callable[[float, float], float]  # (time, temperature) -> Q_gen [W]


@dataclass
class ThermalNode:
    """
    Lumped thermal mass.
    C: Thermal capacitance [J/K]
    hA: Convection coefficient * area to ambient [W/K]
    q_gen_fn: Internal heat generation Q(t, T) [W]
    T: Current temperature [°C] (consistent units assumed)
    """
    name: str
    C: float
    hA: float = 0.0
    q_gen_fn: Optional[HeatGenFn] = None
    T: float = field(default=25.0)

    def heat_generation(self, t: float) -> float:
        if self.q_gen_fn is None:
            return 0.0
        return self.q_gen_fn(t, self.T)


@dataclass
class ConductionLink:
    """
    Conduction path between nodes i and j.
    k_ij: Conduction coefficient [W/K]
    """
    i: int
    j: int
    k_ij: float


class ThermalSystem:
    """
    Network of thermal nodes with:
      - Convection to ambient
      - Conduction between nodes
      - Arbitrary internal heat generation
    """

    def __init__(self, ambient_temp: float):
        self.ambient_temp = ambient_temp
        self.nodes: List[ThermalNode] = []
        self.links: List[ConductionLink] = []

    # ---- topology ----

    def add_node(self, node: ThermalNode) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def add_conduction_link(self, i: int, j: int, k_ij: float) -> None:
        self.links.append(ConductionLink(i, j, k_ij))

    def set_initial_temperatures(self, temps: Dict[str, float]) -> None:
        for node in self.nodes:
            if node.name not in temps:
                raise ValueError(f"Missing initial temperature for node '{node.name}'")
            node.T = temps[node.name]

    # ---- physics ----

    def _compute_dT_dt(self, t: float) -> List[float]:
        """Compute dT/dt for each node at time t."""
        n = len(self.nodes)
        dT_dt = [0.0] * n

        # convection + internal generation
        for i, node in enumerate(self.nodes):
            q_conv = node.hA * (node.T - self.ambient_temp)  # >0 => losing heat
            q_gen = node.heat_generation(t)
            dT_dt[i] += (q_gen - q_conv) / node.C

        # conduction
        for link in self.links:
            Ti = self.nodes[link.i].T
            Tj = self.nodes[link.j].T
            q_i_to_j = link.k_ij * (Ti - Tj)  # +ve when heat flows i -> j
            dT_dt[link.i] += -q_i_to_j / self.nodes[link.i].C
            dT_dt[link.j] +=  q_i_to_j / self.nodes[link.j].C

        return dT_dt

    def step(self, t: float, dt: float) -> None:
        """Advance state by dt using explicit Euler."""
        dT_dt = self._compute_dT_dt(t)
        for i, node in enumerate(self.nodes):
            node.T += dT_dt[i] * dt

    def simulate(self, t_end: float, dt: float) -> Tuple[List[float], Dict[str, List[float]]]:
        """Run simulation and return (times, temps[name] -> list)."""
        times: List[float] = []
        temps: Dict[str, List[float]] = {n.name: [] for n in self.nodes}

        t = 0.0
        while t <= t_end:
            times.append(t)
            for node in self.nodes:
                temps[node.name].append(node.T)
            self.step(t, dt)
            t += dt

        return times, temps


# ---------------------------------------------------------------------------
# Scenario: Motor + Housing (base case)
# ---------------------------------------------------------------------------

def create_motor_system(
    ambient: float = 25.0,
    high_load_w: float = 700.0,
    low_load_w: float = 250.0,
    overload_duration_s: float = 3600.0,
    core_hA: float = 4.0,
    housing_hA: float = 20.0,
) -> ThermalSystem:
    """
    Two-node motor system:
      - motor_core: large mass, internal losses, weak convection
      - housing   : smaller mass, strong convection to ambient
    Parameters exposed to support sensitivity studies.
    """

    system = ThermalSystem(ambient_temp=ambient)

    def motor_q(t: float, T: float) -> float:
        # Overload phase then reduced load
        return high_load_w if t < overload_duration_s else low_load_w

    motor_core = ThermalNode(
        name="motor_core",
        C=6000.0,
        hA=core_hA,
        q_gen_fn=motor_q,
        T=ambient,
    )

    housing = ThermalNode(
        name="housing",
        C=2500.0,
        hA=housing_hA,
        q_gen_fn=None,
        T=ambient,
    )

    idx_core = system.add_node(motor_core)
    idx_housing = system.add_node(housing)

    system.add_conduction_link(idx_core, idx_housing, k_ij=12.0)

    system.set_initial_temperatures({
        "motor_core": ambient,
        "housing": ambient,
    })

    return system


# ---------------------------------------------------------------------------
# Analysis Helpers
# ---------------------------------------------------------------------------

def find_hotspot(times: List[float], temps: Dict[str, List[float]]) -> Tuple[str, float, float]:
    """
    Return (node_name, peak_temp, time_at_peak) for the global hottest point.
    """
    best_node = None
    best_T = float("-inf")
    best_t = 0.0

    for name, series in temps.items():
        for t, T in zip(times, series):
            if T > best_T:
                best_T = T
                best_t = t
                best_node = name

    assert best_node is not None
    return best_node, best_T, best_t


def compute_peak_temp(temps: Dict[str, List[float]], node_name: str) -> float:
    return max(temps[node_name])


# ---------------------------------------------------------------------------
# Nominal Simulation + Hot-Spot Plot
# ---------------------------------------------------------------------------

def run_nominal_sim(failure_threshold: float = 120.0) -> None:
    """
    Run baseline motor scenario and report hot-spot + margin.
    """
    print("=== Nominal Motor Scenario ===")
    system = create_motor_system()

    t_end = 4 * 3600.0   # 4 hours
    dt = 1.0

    times, temps = system.simulate(t_end=t_end, dt=dt)

    node, peak_T, t_peak = find_hotspot(times, temps)
    margin = failure_threshold - peak_T

    print(f"Hot-spot node         : {node}")
    print(f"Peak temperature      : {peak_T:.2f} °C at t = {t_peak/3600:.2f} h")
    print(f"Failure threshold     : {failure_threshold:.2f} °C")
    print(f"Thermal margin        : {margin:.2f} °C")

    if HAS_MPL:
        hours = [t / 3600.0 for t in times]
        plt.figure()
        plt.plot(hours, temps["motor_core"], label="Motor core")
        plt.plot(hours, temps["housing"], label="Housing")
        plt.axhline(system.ambient_temp, linestyle="--", linewidth=0.8, label="Ambient")
        plt.axhline(failure_threshold, linestyle="--", linewidth=0.8, color="r", label="Failure threshold")
        plt.scatter([t_peak/3600.0], [peak_T], color="r", zorder=5, label="Hot-spot peak")
        plt.xlabel("Time [hours]")
        plt.ylabel("Temperature [°C]")
        plt.title("Thermal Response - Nominal Scenario")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Parametric Sensitivity Studies
# ---------------------------------------------------------------------------

def sensitivity_load(t_end: float = 2*3600.0, dt: float = 1.0) -> None:
    print("\n=== Sensitivity: Load vs Peak Core Temperature ===")
    loads = [500.0, 700.0, 900.0, 1100.0]
    peaks = []

    for high_load in loads:
        system = create_motor_system(high_load_w=high_load)
        _, temps = system.simulate(t_end=t_end, dt=dt)
        peak = compute_peak_temp(temps, "motor_core")
        peaks.append(peak)
        print(f"High load {high_load:5.0f} W -> Peak core temp: {peak:6.2f} °C")

    if HAS_MPL:
        plt.figure()
        plt.plot(loads, peaks, marker="o")
        plt.xlabel("High-load power [W]")
        plt.ylabel("Peak motor core temperature [°C]")
        plt.title("Load Sensitivity - Peak Temperature")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def sensitivity_cooling(t_end: float = 2*3600.0, dt: float = 1.0) -> None:
    print("\n=== Sensitivity: Housing Cooling (hA) vs Peak Core Temperature ===")
    hAs = [10.0, 15.0, 20.0, 30.0]
    peaks = []

    for housing_hA in hAs:
        system = create_motor_system(housing_hA=housing_hA)
        _, temps = system.simulate(t_end=t_end, dt=dt)
        peak = compute_peak_temp(temps, "motor_core")
        peaks.append(peak)
        print(f"housing hA = {housing_hA:4.1f} W/K -> Peak core temp: {peak:6.2f} °C")

    if HAS_MPL:
        plt.figure()
        plt.plot(hAs, peaks, marker="o")
        plt.xlabel("Housing hA [W/K]")
        plt.ylabel("Peak motor core temperature [°C]")
        plt.title("Cooling Sensitivity - Peak Temperature")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def sensitivity_ambient(t_end: float = 2*3600.0, dt: float = 1.0) -> None:
    print("\n=== Sensitivity: Ambient Temperature vs Peak Core Temperature ===")
    ambients = [15.0, 25.0, 35.0, 45.0]
    peaks = []

    for ambient in ambients:
        system = create_motor_system(ambient=ambient)
        _, temps = system.simulate(t_end=t_end, dt=dt)
        peak = compute_peak_temp(temps, "motor_core")
        peaks.append(peak)
        print(f"Ambient {ambient:4.1f} °C -> Peak core temp: {peak:6.2f} °C")

    if HAS_MPL:
        plt.figure()
        plt.plot(ambients, peaks, marker="o")
        plt.xlabel("Ambient temperature [°C]")
        plt.ylabel("Peak motor core temperature [°C]")
        plt.title("Ambient Sensitivity - Peak Temperature")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Overload-Duration Study
# ---------------------------------------------------------------------------

def overload_duration_study(
    failure_threshold: float = 120.0,
    t_end: float = 3*3600.0,
    dt: float = 1.0,
) -> None:
    """
    Study how overload duration impacts peak temperature and margin.
    """
    print("\n=== Overload Duration Study ===")
    durations = [15*60.0, 30*60.0, 60*60.0, 90*60.0]  # seconds
    peak_temps = []
    margins = []

    for dur in durations:
        system = create_motor_system(overload_duration_s=dur)
        times, temps = system.simulate(t_end=t_end, dt=dt)
        _, peak_T, _ = find_hotspot(times, temps)
        margin = failure_threshold - peak_T
        peak_temps.append(peak_T)
        margins.append(margin)
        print(f"Overload {dur/60:4.0f} min -> Peak core: {peak_T:6.2f} °C, margin: {margin:6.2f} °C")

    if HAS_MPL:
        mins = [d/60.0 for d in durations]
        plt.figure()
        plt.plot(mins, peak_temps, marker="o", label="Peak core temp")
        plt.axhline(failure_threshold, linestyle="--", color="r", label="Failure threshold")
        plt.xlabel("Overload duration [min]")
        plt.ylabel("Peak core temperature [°C]")
        plt.title("Overload Duration vs Peak Temperature")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    run_nominal_sim()
    sensitivity_load()
    sensitivity_cooling()
    sensitivity_ambient()
    overload_duration_study()


if __name__ == "__main__":
    main()
