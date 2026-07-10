"""Shared pytest configuration for the MHKiT test suite."""

import matplotlib

# Use the non-interactive Agg backend so figure/animation tests run headlessly
# on every OS (Windows otherwise picks TkAgg and fails without Tcl/Tk).
matplotlib.use("Agg")
