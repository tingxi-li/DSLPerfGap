#!/usr/bin/env python3
"""Comprehensive evaluation harness. See evaluation/ package."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from evaluation.__main__ import main
main()
