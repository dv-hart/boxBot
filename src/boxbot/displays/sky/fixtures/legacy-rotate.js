/* generated from the real display spec + placeholder data (dev fixture) */
BB.fixture = {
 "v": 1,
 "display": "rotate",
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
  "gap": 16,
  "padding": 40,
  "children": [
   {
    "type": "text",
    "content": "Rotating",
    "size": "small",
    "color": "muted",
    "weight": "semibold"
   },
   {
    "type": "rotate",
    "source": "",
    "key": "",
    "interval": 5,
    "children": [
     {
      "type": "card",
      "padding": 16,
      "children": [
       {
        "type": "text",
        "content": "First page",
        "size": "title"
       },
       {
        "type": "text",
        "content": "One of three",
        "color": "muted"
       }
      ]
     },
     {
      "type": "card",
      "padding": 16,
      "children": [
       {
        "type": "text",
        "content": "Second page",
        "size": "title"
       }
      ]
     },
     {
      "type": "calendar_widget",
      "data_source": "calendar"
     }
    ]
   },
   {
    "type": "spacer"
   },
   {
    "type": "page_dots"
   }
  ]
 },
 "data": {
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
