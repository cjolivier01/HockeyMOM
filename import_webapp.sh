#!/bin/bash

WEBAPP_URL=http://127.0.0.1:8008

# Authorization: Bearer u57LatsYtQ7rwlpx-84JQmR9v5psYOhEj7y4Zu6TB-Y
# WEB_ACCESS_KEY="--import-token=u57LatsYtQ7rwlpx-84JQmR9v5psYOhEj7y4Zu6TB-Y"

# ./p tools/webapp/import_time2score.py --source=caha --league-name=Norcal --season 0 --config /opt/hm-webapp/app/config.json --user-email cjolivier01@gmail.com
# ./p tools/webapp/import_time2score.py --source=caha --league-name=Norcal --season 0 --config /opt/hm-webapp/app/config.json --user-email cjolivier01@gmail.com --division 6:0

# PROJECT_ID="sage-courier-241217"
# REDEPLOY_WEB="python3 tools/webapp/redeploy_gcp.py --project ${PROJECT_ID} --zone us-central1-a --instance hm-webapp"
# $REDEPLOY_WEB

./tools/webapp/redeploy_local.sh
./p tools/webapp/reset_league_data.py --force
./p tools/webapp/import_time2score.py --source=caha --league-name=Norcal --season 0 --api-url ${WEBAPP_URL} ${WEB_ACCESS_KEY} --user-email cjolivier01@gmail.com --division 6:0
./p scripts/parse_shift_spreadsheet.py --file-list ~/Videos/game_list_long.txt --shifts --upload-webapp --webapp-url=${WEBAPP_URL} ${WEB_ACCESS_KEY} --webapp-owner-email cjolivier01@gmail.com --webapp-league-name=Norcal

