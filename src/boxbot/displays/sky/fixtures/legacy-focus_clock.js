/* generated from the real display spec + placeholder data (dev fixture) */
BB.fixture = {
 "v": 1,
 "display": "focus_clock",
 "now": "2026-11-24T16:02:00-08:00",
 "people": [
  "Jacob",
  "Carina",
  "Erik",
  "Mia"
 ],
 "atmosphere": {
  "phase": "golden",
  "weather": "partly",
  "time": "16:02"
 },
 "layout": {
  "type": "column",
  "padding": [
   0,
   48,
   44,
   48
  ],
  "gap": 0,
  "children": [
   {
    "type": "spacer"
   },
   {
    "type": "clock",
    "size": "xl",
    "format": "12h",
    "show_date": true,
    "show_seconds": false
   },
   {
    "type": "spacer"
   },
   {
    "type": "row",
    "gap": 14,
    "align": "center",
    "children": [
     {
      "type": "icon",
      "name": "cloud-sun",
      "size": "sm",
      "color": "muted"
     },
     {
      "type": "text",
      "content": "72°  Partly Cloudy",
      "size": "caption",
      "color": "muted"
     },
     {
      "type": "text",
      "content": "·",
      "size": "caption",
      "color": "dim"
     },
     {
      "type": "icon",
      "name": "calendar-clock",
      "size": "sm",
      "color": "muted"
     },
     {
      "type": "text",
      "content": "9:00 AM  Team Standup",
      "size": "caption",
      "color": "muted"
     }
    ]
   }
  ]
 },
 "data": {
  "weather": {
   "temp": "72",
   "condition": "Partly Cloudy",
   "icon": "cloud-sun",
   "humidity": "65",
   "wind": "12 mph",
   "forecast": [
    {
     "day": "Mon",
     "icon": "sun",
     "high": "75",
     "low": "58"
    },
    {
     "day": "Tue",
     "icon": "cloud",
     "high": "68",
     "low": "55"
    },
    {
     "day": "Wed",
     "icon": "cloud-rain",
     "high": "62",
     "low": "50"
    },
    {
     "day": "Thu",
     "icon": "sun",
     "high": "70",
     "low": "54"
    },
    {
     "day": "Fri",
     "icon": "cloud-sun",
     "high": "73",
     "low": "57"
    }
   ]
  },
  "calendar": {
   "events": [
    {
     "id": "sample-1",
     "title": "Team Standup",
     "time": "9:00 AM",
     "when": "Today · 9:00 AM",
     "start": "2026-11-25T09:00:00-08:00",
     "end": "2026-11-25T09:30:00-08:00",
     "duration": "30m",
     "location": "Zoom",
     "all_day": false,
     "description": ""
    },
    {
     "id": "sample-2",
     "title": "Design Review",
     "time": "11:00 AM",
     "when": "Today · 11:00 AM",
     "start": "2026-11-25T11:00:00-08:00",
     "end": "2026-11-25T12:00:00-08:00",
     "duration": "1h",
     "location": "Conference Room",
     "all_day": false,
     "description": ""
    },
    {
     "id": "sample-3",
     "title": "Dentist Appointment",
     "time": "2:00 PM",
     "when": "Today · 2:00 PM",
     "start": "2026-11-25T14:00:00-08:00",
     "end": "2026-11-25T15:00:00-08:00",
     "duration": "1h",
     "location": "Downtown",
     "all_day": false,
     "description": ""
    }
   ],
   "count": 3
  }
 },
 "args": {},
 "status": null,
 "presence": null
};
