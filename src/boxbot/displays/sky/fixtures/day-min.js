/* day-min — specs/day.json with its bindings resolved against the sample data below (Tue 24 Nov 2026, Seattle).
   Only real fields: no event names a person, so there is one shared column; nothing already past.
   Regenerate rather than hand-edit if the spec changes: layout must stay what Python would push. */
BB.fixture = {
 "v": 1,
 "display": "day",
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
      "range": "day",
      "events": [
       {
        "id": "evt05",
        "title": "Lunch with Dana",
        "time": "12:15 PM",
        "when": "Today · 12:15 PM",
        "start": "2026-11-24T12:15:00-08:00",
        "end": "2026-11-24T13:15:00-08:00",
        "location": "Ballard",
        "duration": "1h",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt06",
        "title": "School pickup",
        "time": "3:10 PM",
        "when": "Today · 3:10 PM",
        "start": "2026-11-24T15:10:00-08:00",
        "end": "2026-11-24T15:30:00-08:00",
        "location": "early release",
        "duration": "20m",
        "all_day": false,
        "description": ""
       },
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
     }
    ]
   },
   {
    "type": "zone",
    "place": "sky-right",
    "children": [
     {
      "type": "now",
      "style": "compact",
      "weather": {
       "temp": "48",
       "condition": "Sunny",
       "icon": "sun",
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
    "type": "surface",
    "dock": "bottom",
    "children": [
     {
      "type": "day_grid",
      "events": [
       {
        "id": "evt05",
        "title": "Lunch with Dana",
        "time": "12:15 PM",
        "when": "Today · 12:15 PM",
        "start": "2026-11-24T12:15:00-08:00",
        "end": "2026-11-24T13:15:00-08:00",
        "location": "Ballard",
        "duration": "1h",
        "all_day": false,
        "description": ""
       },
       {
        "id": "evt06",
        "title": "School pickup",
        "time": "3:10 PM",
        "when": "Today · 3:10 PM",
        "start": "2026-11-24T15:10:00-08:00",
        "end": "2026-11-24T15:30:00-08:00",
        "location": "early release",
        "duration": "20m",
        "all_day": false,
        "description": ""
       },
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
      "from": "07:30",
      "to": "20:30",
      "background": []
     }
    ]
   }
  ]
 },
 "data": {
  "weather": {
   "temp": "48",
   "condition": "Sunny",
   "icon": "sun",
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
     "id": "evt05",
     "title": "Lunch with Dana",
     "time": "12:15 PM",
     "when": "Today · 12:15 PM",
     "start": "2026-11-24T12:15:00-08:00",
     "end": "2026-11-24T13:15:00-08:00",
     "location": "Ballard",
     "duration": "1h",
     "all_day": false,
     "description": ""
    },
    {
     "id": "evt06",
     "title": "School pickup",
     "time": "3:10 PM",
     "when": "Today · 3:10 PM",
     "start": "2026-11-24T15:10:00-08:00",
     "end": "2026-11-24T15:30:00-08:00",
     "location": "early release",
     "duration": "20m",
     "all_day": false,
     "description": ""
    },
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
   "count": 19
  },
  "day": {
   "background": []
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
