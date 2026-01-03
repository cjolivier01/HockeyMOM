#!/bin/bash
./p tools/webapp/reset_league_data.py
./p tools/webapp/import_time2score.py --source=caha --league-name=Norcal --season 0 --config /opt/hm-webapp/app/config.json --user-email cjolivier01@gmail.com --division 6:0
# ./p tools/webapp/import_time2score.py --source=caha --league-name=Norcal --season 0 --api-url http://127.0.0.1:8008 --user-email cjolivier01@gmail.com --division 6:0
./p scripts/parse_shift_spreadsheet.py --file-list ~/Videos/game_list_long.txt --shifts --upload-webapp --webapp-url=http://127.0.0.1:8008
./p tools/webapp/dedupe_league_teams.py --config /opt/hm-webapp/app/config.json --league-name Norcal --yes
