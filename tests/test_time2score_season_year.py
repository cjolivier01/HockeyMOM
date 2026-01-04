from __future__ import annotations

import datetime as dt

from hmlib.time2score import util


def should_infer_season_year_sept_to_dec_as_start_year_and_jan_to_jun_as_next_year():
    ref = dt.datetime(2026, 1, 3, 12, 0, 0)
    # Treat the active season as Oct-Jun; Jul-Sep are offseason/preseason and
    # should map to the season start year.
    assert util.infer_season_year_for_date_str("Sat Sep 13", reference_dt=ref) == 2025
    assert util.infer_season_year_for_date_str("Sat Aug 29", reference_dt=ref) == 2025
    assert util.infer_season_year_for_date_str("Mon Dec 29", reference_dt=ref) == 2025
    assert util.infer_season_year_for_date_str("Sat Jan 03", reference_dt=ref) == 2026
    assert util.infer_season_year_for_date_str("Fri Jun 19", reference_dt=ref) == 2026
