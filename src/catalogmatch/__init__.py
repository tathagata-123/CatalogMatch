"""Foundations for product entity matching."""

__version__ = "0.1.0"

from catalogmatch.pipeline import train_bundle, save_bundle, load_bundle, match_record

__all__ = ["train_bundle", "save_bundle", "load_bundle", "match_record"]
