sbts_advanced/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── bandwidth.py        # Phase 1: Adaptive CV Bandwidth
│   ├── lightsb.py          # Phase 2: GMM Parameterization
│   ├── reference.py        # Phase 3/4: Stoch Vol & Levy Processes
│   └── solver.py           # Euler-Maruyama with Jumps
├── models/
│   ├── __init__.py
│   ├── jumps.py            # Jump Detection (Ait-Sahalia)
│   └── calibration.py      # Volatility Surface & Calibration
├── utils/
│   ├── __init__.py
│   └── metrics.py          # MSE, Wasserstein, etc.
└── main.py                 # Example pipeline usage