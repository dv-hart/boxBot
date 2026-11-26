/* home-min — specs/home.json with its bindings resolved against the sample data below (Tue 24 Nov 2026, Seattle).
   Only what Python has today: NOAA weather (no outlook), bare Google events, no sun times, no reply count.
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
      "weather": {
       "temp": "48",
       "condition": "Partly Sunny",
       "icon": "cloud-sun",
       "humidity": "65",
       "wind": "12 mph",
       "forecast": [
        {
         "day": "Tue",
         "icon": "cloud-sun",
         "high": "48",
         "low": "41",
         "condition": "Sunbreaks"
        },
        {
         "day": "Wed",
         "icon": "cloud-rain",
         "high": "46",
         "low": "41",
         "condition": "Rain"
        },
        {
         "day": "Thu",
         "icon": "cloudy",
         "high": "45",
         "low": "38",
         "condition": "Cloudy"
        },
        {
         "day": "Fri",
         "icon": "sun",
         "high": "44",
         "low": "34",
         "condition": "Sunny"
        },
        {
         "day": "Sat",
         "icon": "cloud-rain",
         "high": "47",
         "low": "40",
         "condition": "Rain Likely"
        },
        {
         "day": "Sun",
         "icon": "cloud-sun",
         "high": "46",
         "low": "39",
         "condition": "Partly Sunny"
        },
        {
         "day": "Mon",
         "icon": "cloud-rain",
         "high": "45",
         "low": "40",
         "condition": "Rain"
        }
       ],
       "updated_at": "2026-11-24T15:52:00-08:00"
      }
     }
    ]
   },
   {
    "type": "zone",
    "place": "ground",
    "children": [
     {
      "type": "next_event",
      "events": [
       {
        "id": "evt07",
        "title": "Swim practice",
        "time": "5:30 PM",
        "when": "Today · 5:30 PM",
        "start": "2026-11-24T17:30:00-08:00",
        "end": "2026-11-24T18:30:00-08:00",
        "location": "Green Lake pool",
        "duration": "1h",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt08",
        "title": "Parent–teacher conference",
        "time": "7:00 PM",
        "when": "Today · 7:00 PM",
        "start": "2026-11-24T19:00:00-08:00",
        "end": "2026-11-24T19:40:00-08:00",
        "location": "Bryant Elementary",
        "duration": "40m",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt12",
        "title": "No school",
        "time": "all day",
        "when": "all day",
        "start": "2026-11-25T00:00:00",
        "end": "2026-11-28T00:00:00",
        "location": "",
        "duration": "",
        "all_day": true,
        "description": ""
       },
       {
        "id": "evt09",
        "title": "Ferry to Bainbridge",
        "time": "8:10 AM",
        "when": "Tomorrow · 8:10 AM",
        "start": "2026-11-25T08:10:00-08:00",
        "end": "2026-11-25T08:45:00-08:00",
        "location": "Colman Dock",
        "duration": "35m",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt10",
        "title": "Piano lesson",
        "time": "4:00 PM",
        "when": "Tomorrow · 4:00 PM",
        "start": "2026-11-25T16:00:00-08:00",
        "end": "2026-11-25T16:45:00-08:00",
        "location": "Ravenna",
        "duration": "45m",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt11",
        "title": "Pie baking",
        "time": "6:00 PM",
        "when": "Tomorrow · 6:00 PM",
        "start": "2026-11-25T18:00:00-08:00",
        "end": "2026-11-25T19:30:00-08:00",
        "location": "home",
        "duration": "1h 30m",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt16",
        "title": "Grandma June visiting",
        "time": "all day",
        "when": "all day",
        "start": "2026-11-26T00:00:00",
        "end": "2026-11-30T00:00:00",
        "location": "",
        "duration": "",
        "all_day": true,
        "description": ""
       },
       {
        "id": "evt14",
        "title": "Turkey trot 5K",
        "time": "9:00 AM",
        "when": "Thu Nov 26 · 9:00 AM",
        "start": "2026-11-26T09:00:00-08:00",
        "end": "2026-11-26T10:00:00-08:00",
        "location": "Green Lake",
        "duration": "1h",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt15",
        "title": "Thanksgiving dinner",
        "time": "3:00 PM",
        "when": "Thu Nov 26 · 3:00 PM",
        "start": "2026-11-26T15:00:00-08:00",
        "end": "2026-11-26T18:00:00-08:00",
        "location": "home",
        "duration": "3h",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt17",
        "title": "Soccer vs. Ballard",
        "time": "10:00 AM",
        "when": "Sat Nov 28 · 10:00 AM",
        "start": "2026-11-28T10:00:00-08:00",
        "end": "2026-11-28T11:30:00-08:00",
        "location": "Magnuson Park",
        "duration": "1h 30m",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt18",
        "title": "Ballet",
        "time": "10:00 AM",
        "when": "Sat Nov 28 · 10:00 AM",
        "start": "2026-11-28T10:00:00-08:00",
        "end": "2026-11-28T11:00:00-08:00",
        "location": "Capitol Hill",
        "duration": "1h",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt19",
        "title": "Pike Place with Grandma",
        "time": "2:00 PM",
        "when": "Sat Nov 28 · 2:00 PM",
        "start": "2026-11-28T14:00:00-08:00",
        "end": "2026-11-28T16:00:00-08:00",
        "location": "Pike Place",
        "duration": "2h",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt20",
        "title": "Dinner out",
        "time": "7:00 PM",
        "when": "Sat Nov 28 · 7:00 PM",
        "start": "2026-11-28T19:00:00-08:00",
        "end": "2026-11-28T21:30:00-08:00",
        "location": "Ballard",
        "duration": "2h 30m",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt21",
        "title": "Brunch",
        "time": "11:00 AM",
        "when": "Sun Nov 29 · 11:00 AM",
        "start": "2026-11-29T11:00:00-08:00",
        "end": "2026-11-29T12:30:00-08:00",
        "location": "home",
        "duration": "1h 30m",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt22",
        "title": "Grandma’s flight",
        "time": "4:35 PM",
        "when": "Sun Nov 29 · 4:35 PM",
        "start": "2026-11-29T16:35:00-08:00",
        "end": "2026-11-29T17:30:00-08:00",
        "location": "Sea-Tac",
        "duration": "55m",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt23",
        "title": "Orthodontist",
        "time": "8:30 AM",
        "when": "Mon Nov 30 · 8:30 AM",
        "start": "2026-11-30T08:30:00-08:00",
        "end": "2026-11-30T09:10:00-08:00",
        "location": "Northgate",
        "duration": "40m",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt13",
        "title": "Thanksgiving",
        "time": "all day",
        "when": "all day",
        "start": "2026-11-26T00:00:00",
        "end": "2026-11-27T00:00:00",
        "location": "",
        "duration": "",
        "all_day": true,
        "description": ""
       }
      ]
     },
     {
      "type": "house",
      "present": [
       {
        "name": "Jacob",
        "since": "2026-11-24T15:20:00-08:00"
       },
       {
        "name": "Carina",
        "since": "2026-11-24T15:20:00-08:00"
       },
       {
        "name": "Erik",
        "since": "2026-11-24T15:20:00-08:00"
       },
       {
        "name": "Mia",
        "since": "2026-11-24T15:20:00-08:00"
       }
      ],
      "replies": 0,
      "todos": [
       {
        "id": "t1",
        "description": "Sign Erik’s swim-meet form",
        "due_date": "2026-11-24",
        "for_person": "Jacob",
        "status": "pending"
       },
       {
        "id": "t2",
        "description": "Pick up the turkey",
        "due_date": "2026-11-25",
        "for_person": "Carina",
        "status": "pending"
       },
       {
        "id": "t3",
        "description": "Call the plumber back",
        "due_date": null,
        "for_person": "Jacob",
        "status": "pending"
       },
       {
        "id": "t4",
        "description": "Renew the car tabs",
        "due_date": "2026-12-04",
        "for_person": null,
        "status": "pending"
       }
      ]
     }
    ]
   }
  ]
 },
 "data": {
  "weather": {
   "temp": "48",
   "condition": "Partly Sunny",
   "icon": "cloud-sun",
   "humidity": "65",
   "wind": "12 mph",
   "forecast": [
    {
     "day": "Tue",
     "icon": "cloud-sun",
     "high": "48",
     "low": "41",
     "condition": "Sunbreaks"
    },
    {
     "day": "Wed",
     "icon": "cloud-rain",
     "high": "46",
     "low": "41",
     "condition": "Rain"
    },
    {
     "day": "Thu",
     "icon": "cloudy",
     "high": "45",
     "low": "38",
     "condition": "Cloudy"
    },
    {
     "day": "Fri",
     "icon": "sun",
     "high": "44",
     "low": "34",
     "condition": "Sunny"
    },
    {
     "day": "Sat",
     "icon": "cloud-rain",
     "high": "47",
     "low": "40",
     "condition": "Rain Likely"
    },
    {
     "day": "Sun",
     "icon": "cloud-sun",
     "high": "46",
     "low": "39",
     "condition": "Partly Sunny"
    },
    {
     "day": "Mon",
     "icon": "cloud-rain",
     "high": "45",
     "low": "40",
     "condition": "Rain"
    }
   ],
   "updated_at": "2026-11-24T15:52:00-08:00"
  },
  "calendar": {
   "events": [
    {
     "id": "evt07",
     "title": "Swim practice",
     "time": "5:30 PM",
     "when": "Today · 5:30 PM",
     "start": "2026-11-24T17:30:00-08:00",
     "end": "2026-11-24T18:30:00-08:00",
     "location": "Green Lake pool",
     "duration": "1h",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt08",
     "title": "Parent–teacher conference",
     "time": "7:00 PM",
     "when": "Today · 7:00 PM",
     "start": "2026-11-24T19:00:00-08:00",
     "end": "2026-11-24T19:40:00-08:00",
     "location": "Bryant Elementary",
     "duration": "40m",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt12",
     "title": "No school",
     "time": "all day",
     "when": "all day",
     "start": "2026-11-25T00:00:00",
     "end": "2026-11-28T00:00:00",
     "location": "",
     "duration": "",
     "all_day": true,
     "description": ""
    },
    {
     "id": "evt09",
     "title": "Ferry to Bainbridge",
     "time": "8:10 AM",
     "when": "Tomorrow · 8:10 AM",
     "start": "2026-11-25T08:10:00-08:00",
     "end": "2026-11-25T08:45:00-08:00",
     "location": "Colman Dock",
     "duration": "35m",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt10",
     "title": "Piano lesson",
     "time": "4:00 PM",
     "when": "Tomorrow · 4:00 PM",
     "start": "2026-11-25T16:00:00-08:00",
     "end": "2026-11-25T16:45:00-08:00",
     "location": "Ravenna",
     "duration": "45m",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt11",
     "title": "Pie baking",
     "time": "6:00 PM",
     "when": "Tomorrow · 6:00 PM",
     "start": "2026-11-25T18:00:00-08:00",
     "end": "2026-11-25T19:30:00-08:00",
     "location": "home",
     "duration": "1h 30m",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt16",
     "title": "Grandma June visiting",
     "time": "all day",
     "when": "all day",
     "start": "2026-11-26T00:00:00",
     "end": "2026-11-30T00:00:00",
     "location": "",
     "duration": "",
     "all_day": true,
     "description": ""
    },
    {
     "id": "evt14",
     "title": "Turkey trot 5K",
     "time": "9:00 AM",
     "when": "Thu Nov 26 · 9:00 AM",
     "start": "2026-11-26T09:00:00-08:00",
     "end": "2026-11-26T10:00:00-08:00",
     "location": "Green Lake",
     "duration": "1h",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt15",
     "title": "Thanksgiving dinner",
     "time": "3:00 PM",
     "when": "Thu Nov 26 · 3:00 PM",
     "start": "2026-11-26T15:00:00-08:00",
     "end": "2026-11-26T18:00:00-08:00",
     "location": "home",
     "duration": "3h",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt17",
     "title": "Soccer vs. Ballard",
     "time": "10:00 AM",
     "when": "Sat Nov 28 · 10:00 AM",
     "start": "2026-11-28T10:00:00-08:00",
     "end": "2026-11-28T11:30:00-08:00",
     "location": "Magnuson Park",
     "duration": "1h 30m",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt18",
     "title": "Ballet",
     "time": "10:00 AM",
     "when": "Sat Nov 28 · 10:00 AM",
     "start": "2026-11-28T10:00:00-08:00",
     "end": "2026-11-28T11:00:00-08:00",
     "location": "Capitol Hill",
     "duration": "1h",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt19",
     "title": "Pike Place with Grandma",
     "time": "2:00 PM",
     "when": "Sat Nov 28 · 2:00 PM",
     "start": "2026-11-28T14:00:00-08:00",
     "end": "2026-11-28T16:00:00-08:00",
     "location": "Pike Place",
     "duration": "2h",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt20",
     "title": "Dinner out",
     "time": "7:00 PM",
     "when": "Sat Nov 28 · 7:00 PM",
     "start": "2026-11-28T19:00:00-08:00",
     "end": "2026-11-28T21:30:00-08:00",
     "location": "Ballard",
     "duration": "2h 30m",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt21",
     "title": "Brunch",
     "time": "11:00 AM",
     "when": "Sun Nov 29 · 11:00 AM",
     "start": "2026-11-29T11:00:00-08:00",
     "end": "2026-11-29T12:30:00-08:00",
     "location": "home",
     "duration": "1h 30m",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt22",
     "title": "Grandma’s flight",
     "time": "4:35 PM",
     "when": "Sun Nov 29 · 4:35 PM",
     "start": "2026-11-29T16:35:00-08:00",
     "end": "2026-11-29T17:30:00-08:00",
     "location": "Sea-Tac",
     "duration": "55m",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt23",
     "title": "Orthodontist",
     "time": "8:30 AM",
     "when": "Mon Nov 30 · 8:30 AM",
     "start": "2026-11-30T08:30:00-08:00",
     "end": "2026-11-30T09:10:00-08:00",
     "location": "Northgate",
     "duration": "40m",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt13",
     "title": "Thanksgiving",
     "time": "all day",
     "when": "all day",
     "start": "2026-11-26T00:00:00",
     "end": "2026-11-27T00:00:00",
     "location": "",
     "duration": "",
     "all_day": true,
     "description": ""
    }
   ],
   "count": 17
  },
  "tasks": {
   "items": [
    {
     "id": "t1",
     "description": "Sign Erik’s swim-meet form",
     "due_date": "2026-11-24",
     "for_person": "Jacob",
     "status": "pending"
    },
    {
     "id": "t2",
     "description": "Pick up the turkey",
     "due_date": "2026-11-25",
     "for_person": "Carina",
     "status": "pending"
    },
    {
     "id": "t3",
     "description": "Call the plumber back",
     "due_date": null,
     "for_person": "Jacob",
     "status": "pending"
    },
    {
     "id": "t4",
     "description": "Renew the car tabs",
     "due_date": "2026-12-04",
     "for_person": null,
     "status": "pending"
    }
   ],
   "count": 4
  },
  "people": {
   "present": [
    {
     "name": "Jacob",
     "since": "2026-11-24T15:20:00-08:00"
    },
    {
     "name": "Carina",
     "since": "2026-11-24T15:20:00-08:00"
    },
    {
     "name": "Erik",
     "since": "2026-11-24T15:20:00-08:00"
    },
    {
     "name": "Mia",
     "since": "2026-11-24T15:20:00-08:00"
    }
   ],
   "count": 4
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
