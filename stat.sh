#!/bin/bash
STAT_FILE="$HOME/RVideos/dh-tv-12-2/stats/dh-tv-12-2.xls"
# STAT_FILE="$HOME/Videos/dh-tv-12-1/stats/dh-tv-12-1.xls"

# TV-12-1
# GOALS_FOR="-g GA:1/14:24 -g GA:1/13:49 -g GA:1/12:28 -g GA:1/4:14 -g GA:2/7:17 -g GA:2/3:04 -g GA:3/14:16 -g GA:3/11:35 -g GA:3/11:07"
# GOALS_AGAINST="-g GF:1/02:53 -g GF:2/6:00"

# TV 12-2
GOALS_FOR="-g GF:2/11:32 -g GF:2/1:35 -g GF:3/4:47 -g GF:3/0:29"
GOALS_AGAINST="-g GA:1/11:53 -g GA:1/7:12 -g GA:2/9:19 -g GA:2/3:36 -g GA:3/9:46 -g GA:3/3:57"

python scripts/parse_shift_spreadsheet.py --input "${STAT_FILE}" ${GOALS_FOR} ${GOALS_AGAINST}
