/* home-empty — specs/home.json with its bindings resolved against the sample data below (Tue 24 Nov 2026, Seattle).
   Sparse: no events, nobody seen, weather fetch failed, no sun times.
   Regenerate rather than hand-edit if the spec changes: layout must stay what Python would push. */
BB.fixture = {
 "v": 1,
 "display": "home",
 "transition": "crossfade",
 "people": [
  "Jacob",
  "Carina",
  "Erik",
  "Mia"
 ],
 "now": "2026-11-24T16:02:00-08:00",
 "atmosphere": {
  "phase": "golden",
  "weather": "partly",
  "time": "16:02",
  "dim": false,
  "sun": null
 },
 "scene": {
  "horizon": "high"
 },
 "layout": {
  "type": "zones",
  "children": [
   {
    "type": "zone",
    "place": "sky",
    "children": [
     {
      "type": "sky_clock",
      "size": "hero"
     },
     {
      "type": "now",
      "weather": {}
     }
    ]
   },
   {
    "type": "zone",
    "place": "ground",
    "children": [
     {
      "type": "next_event",
      "events": []
     },
     {
      "type": "house",
      "present": [],
      "replies": 0,
      "todos": []
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
  "inbox": {
   "needs_reply": 0
  },
  "clock": {
   "hour": 16,
   "minute": 2,
   "second": 0,
   "display": "4:02",
   "date": "November 24, 2026",
   "date_short": "November 24",
   "day_of_week": "Tuesday"
  }
 },
 "args": {},
 "status": null,
 "presence": null
};
