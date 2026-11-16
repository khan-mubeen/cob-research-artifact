"""v5 bidding: prices and prompts for records that carry dataset_version v5

bidding_formulas.py and slm.py dispatch here, and only here, when a profile, rfq or order
record names v5. v4 records carry no dataset_version, so every v4 decision, price and prompt
still runs through the untouched v4 code and never imports this package

  pricing.py   the v5 price terms, logistics eligibility and chain budget, plus the one
               reading of every input they use
  prompts.py   the v5 stage and chain-selection prompts, built from those same inputs
"""
