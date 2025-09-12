#!/bin/bash
STAT_FILE="$HOME/Videos/dh-tv-12-1/stats/dh-tv-12-1.xls"

# GOALS_FOR="-g GA:1/14:24 -g GA:1/13:49 -g GA:1/12:28 -g GA:1/4:14 -g GA:2/7:17 -g GA:2/3:04 -g GA:3/14:16 -g GA:3/11:35 -g GA:3/11:07"
# GOALS_AGAINST="-g GF:1/02:53 -g GF:2/6:00"

GOALS_FOR="-g GF:1/1:04 -g GF:1/00:34 -g GF:2/6:39 -g GF:3/12:18 -g GF:3/9:58 -g GF:3/9:15"
GOALS_AGAINST="-g GA:3/4:29 -g GA:3/2:35"

python scripts/parse_shift_spreadsheet.py --input "${STAT_FILE}" ${GOALS_FOR} ${GOALS_AGAINST}
