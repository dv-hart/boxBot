/* week-empty — specs/week.json with its bindings resolved against the sample data below (Tue 24 Nov 2026, Seattle).
   Sparse: nothing on the calendar, no forecast.
   Regenerate rather than hand-edit if the spec changes: layout must stay what Python would push. */
BB.fixture = {
 "v": 1,
 "display": "week",
 "transition": "crossfade",
 "people": [
  "Jacob",
  "Carina",
  "Erik",
  "Mia"
 ],
 "now": "2026-11-24T12:04:00-08:00",
 "atmosphere": {
  "phase": "noon",
  "weather": "clear",
  "time": "12:04",
  "dim": false,
  "sun": null
 },
 "scene": {
  "horizon": "high",
  "seed": 21
 },
 "layout": {
  "type": "zones",
  "children": [
   {
    "type": "zone",
    "place": "sky",
    "children": [
     {
      "type": "cal_header",
      "range": "week",
      "events": [],
      "start": null,
      "text": ""
     }
    ]
   },
   {
    "type": "zone",
    "place": "sky-right",
    "children": [
     {
      "type": "legend"
     }
    ]
   },
   {
    "type": "surface",
    "dock": "bottom",
    "children": [
     {
      "type": "week_grid",
      "events": [],
      "forecast": null,
      "start": null,
      "max_per_day": 5
     }
    ]
   }
  ]
 },
 "data": {
  "weather": {},
  "calendar": {
   "events": [],
   "count": 0
  },
  "week": {
   "summary": ""
  },
  "clock": {
   "hour": 12,
   "minute": 4,
   "second": 0,
   "display": "12:04",
   "date": "November 24, 2026",
   "date_short": "November 24",
   "day_of_week": "Tuesday"
  }
 },
 "args": {},
 "status": null,
 "presence": null
};
