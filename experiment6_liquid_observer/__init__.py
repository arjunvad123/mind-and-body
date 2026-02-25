"""
Experiment 6: Liquid Neural Network Observer-Executor

A CfC (Closed-form Continuous-time) observer watches a CfC executor
trained on diverse dynamical systems. Both operate in continuous time
via ODEs, enabling probes for consciousness-like properties that
leverage the shared temporal medium.

Key improvements over Experiment 5:
- Continuous-time observation (no discrete snapshots)
- Surprise detection via mathematically precise "interesting moments"
- Tiny networks (50-200 neurons) enabling phase portrait / Lyapunov / Phi analysis
- True cross-executor control (two executors with different seeds)
- 11 probes (6 original adapted + 5 LNN-specific)
"""
