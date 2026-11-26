/* brief-empty — specs/brief.json with its bindings resolved against the sample data below (Tue 24 Nov 2026, Seattle).
   Sparse: nothing on the calendar, no to-dos, weather missing.
   Regenerate rather than hand-edit if the spec changes: layout must stay what Python would push. */
BB.fixture = {
 "v": 1,
 "display": "brief",
 "transition": "crossfade",
 "people": [
  "Jacob",
  "Carina",
  "Erik",
  "Mia"
 ],
 "now": "2026-11-24T07:12:00-08:00",
 "atmosphere": {
  "phase": "dawn",
  "weather": "fog",
  "time": "07:12",
  "dim": false,
  "sun": null
 },
 "scene": {
  "horizon": "low",
  "seed": 12
 },
 "layout": {
  "type": "zones",
  "children": [
   {
    "type": "surface",
    "dock": "left",
    "children": [
     {
      "type": "agenda_timeline",
      "events": [],
      "mode": null,
      "forecast": null,
      "gap_min": 90
     }
    ]
   },
   {
    "type": "zone",
    "place": "sky",
    "children": [
     {
      "type": "sky_clock",
      "size": "compact",
      "weather": {}
     },
     {
      "type": "say",
      "greeting": "",
      "text": "",
      "events": [],
      "who": null
     }
    ]
   },
   {
    "type": "zone",
    "place": "ground",
    "children": [
     {
      "type": "todo_list",
      "items": [],
      "max": 3
     },
     {
      "type": "forecast_strip",
      "days": null,
      "count": 5
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
  "tasks": {
   "items": [],
   "count": 0
  },
  "people": {
   "present": [],
   "count": 0
  },
  "brief": {
   "greeting": "",
   "summary": ""
  },
  "clock": {
   "hour": 7,
   "minute": 12,
   "second": 0,
   "display": "7:12",
   "date": "November 24, 2026",
   "date_short": "November 24",
   "day_of_week": "Tuesday"
  }
 },
 "args": {},
 "status": null,
 "presence": null
};
