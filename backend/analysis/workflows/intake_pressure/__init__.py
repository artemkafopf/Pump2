"""Pump intake pressure (Рприем) reconstruction and the gas fraction that rides on it.

Two *independent* realizations of the same target — a grey-box physical model
(:mod:`analysis.features.intake_pressure`) and a gradient-boosted one
(:mod:`analysis.models.ml.intake_pressure_ml`) — plus the hybrid that nests the first
inside the second.  See :mod:`analysis.workflows.intake_pressure.data` for what the
warehouse actually supplies and why the gap is much smaller than it first looks.
"""
