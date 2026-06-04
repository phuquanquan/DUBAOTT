How to use the dashboard:

1. Refresh data and generate payload:
   D:/DEV/DUBAOTT/.venv/Scripts/python.exe D:/DEV/DUBAOTT/lotto_scraper.py refresh-dashboard --csv D:/DEV/DUBAOTT/xsmb_full.csv --json D:/DEV/DUBAOTT/xsmb_full.json --dashboard-output D:/DEV/DUBAOTT/dashboard/dashboard-payload.json --top-k 3 --pretty

2. Serve the dashboard folder:
   cd D:/DEV/DUBAOTT/dashboard
   D:/DEV/DUBAOTT/.venv/Scripts/python.exe -m http.server 8080

3. Open http://localhost:8080 and click "Load payload".

Notes:
- If dashboard-payload.json is missing, the page falls back to sample data.
- refresh-dashboard crawls only dates missing after the latest date already present in xsmb_full.csv.
- By default, refresh-dashboard treats 18:30 as the daily draw cutoff. Before 18:30 it refreshes only through yesterday; after 18:30 it includes today.
- If the latest draw has not been posted yet even after the cutoff, refresh-dashboard reports it in failed_dates and keeps the existing dataset unchanged.
- Serving over localhost avoids browser restrictions around fetching local JSON files.
