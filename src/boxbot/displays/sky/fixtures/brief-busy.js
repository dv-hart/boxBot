/* brief-busy — specs/brief.json with its bindings resolved against the sample data below (Tue 24 Nov 2026, Seattle).
   Overflow test: fourteen events today. The rail folds past rows into “N done” and later rows into “+N more”. Try &now=2026-11-24T18:15:00-08:00&phase=night.
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
 "now": "2026-11-24T12:40:00-08:00",
 "atmosphere": {
  "phase": "noon",
  "weather": "overcast",
  "time": "12:40",
  "dim": false,
  "sun": {
   "next": "sunset",
   "at": "4:24 PM"
  }
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
      "events": [
       {
        "id": "evt24",
        "title": "Walk the dog",
        "time": "7:30 AM",
        "when": "Today · 7:30 AM",
        "start": "2026-11-24T07:30:00-08:00",
        "end": "2026-11-24T07:55:00-08:00",
        "location": "Ravenna Park",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
       },
       {
        "id": "evt25",
        "title": "School drop-off",
        "time": "8:05 AM",
        "when": "Today · 8:05 AM",
        "start": "2026-11-24T08:05:00-08:00",
        "end": "2026-11-24T08:30:00-08:00",
        "location": "Erik and Mia",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
       },
       {
        "id": "evt26",
        "title": "Design review",
        "time": "9:00 AM",
        "when": "Today · 9:00 AM",
        "start": "2026-11-24T09:00:00-08:00",
        "end": "2026-11-24T09:25:00-08:00",
        "location": "video call",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
       },
       {
        "id": "evt27",
        "title": "Plumber visit",
        "time": "10:00 AM",
        "when": "Today · 10:00 AM",
        "start": "2026-11-24T10:00:00-08:00",
        "end": "2026-11-24T10:25:00-08:00",
        "location": "home",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Carina"
       },
       {
        "id": "evt28",
        "title": "Grocery pickup",
        "time": "11:00 AM",
        "when": "Today · 11:00 AM",
        "start": "2026-11-24T11:00:00-08:00",
        "end": "2026-11-24T11:25:00-08:00",
        "location": "PCC",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Carina"
       },
       {
        "id": "evt29",
        "title": "Lunch with Dana",
        "time": "12:15 PM",
        "when": "Today · 12:15 PM",
        "start": "2026-11-24T12:15:00-08:00",
        "end": "2026-11-24T12:40:00-08:00",
        "location": "Ballard",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Carina"
       },
       {
        "id": "evt30",
        "title": "Quarterly planning with the whole product group",
        "time": "1:30 PM",
        "when": "Today · 1:30 PM",
        "start": "2026-11-24T13:30:00-08:00",
        "end": "2026-11-24T13:55:00-08:00",
        "location": "video call",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
       },
       {
        "id": "evt31",
        "title": "Library returns",
        "time": "2:30 PM",
        "when": "Today · 2:30 PM",
        "start": "2026-11-24T14:30:00-08:00",
        "end": "2026-11-24T14:55:00-08:00",
        "location": "Northeast branch",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Mia"
       },
       {
        "id": "evt32",
        "title": "School pickup",
        "time": "3:10 PM",
        "when": "Today · 3:10 PM",
        "start": "2026-11-24T15:10:00-08:00",
        "end": "2026-11-24T15:35:00-08:00",
        "location": "early release",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Carina"
       },
       {
        "id": "evt33",
        "title": "Piano lesson",
        "time": "4:00 PM",
        "when": "Today · 4:00 PM",
        "start": "2026-11-24T16:00:00-08:00",
        "end": "2026-11-24T16:25:00-08:00",
        "location": "Ravenna",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Mia"
       },
       {
        "id": "evt34",
        "title": "Swim practice",
        "time": "5:30 PM",
        "when": "Today · 5:30 PM",
        "start": "2026-11-24T17:30:00-08:00",
        "end": "2026-11-24T17:55:00-08:00",
        "location": "Green Lake pool",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Erik",
        "leave_by": "5:10 PM"
       },
       {
        "id": "evt35",
        "title": "Dinner",
        "time": "6:30 PM",
        "when": "Today · 6:30 PM",
        "start": "2026-11-24T18:30:00-08:00",
        "end": "2026-11-24T18:55:00-08:00",
        "location": "home",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "family"
       },
       {
        "id": "evt36",
        "title": "Parent–teacher conference",
        "time": "7:00 PM",
        "when": "Today · 7:00 PM",
        "start": "2026-11-24T19:00:00-08:00",
        "end": "2026-11-24T19:25:00-08:00",
        "location": "Bryant Elementary",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "family"
       },
       {
        "id": "evt37",
        "title": "Bins to the curb",
        "time": "8:30 PM",
        "when": "Today · 8:30 PM",
        "start": "2026-11-24T20:30:00-08:00",
        "end": "2026-11-24T20:55:00-08:00",
        "location": "",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
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
        "description": "",
        "people": [
         "Erik",
         "Mia"
        ]
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
        "description": "",
        "person": "family",
        "leave_by": "2026-11-25T07:25:00-08:00",
        "short_title": "Ferry"
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
        "description": "",
        "person": "Mia",
        "short_title": "Piano"
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
        "description": "",
        "person": "Carina"
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
        "description": "",
        "kind": "holiday"
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
        "description": "",
        "person": "family",
        "short_title": "Grandma June"
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
        "description": "",
        "person": "Jacob",
        "short_title": "Turkey trot"
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
        "description": "",
        "person": "family",
        "short_title": "Dinner at 3"
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
        "description": "",
        "person": "Erik",
        "short_title": "Soccer"
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
        "description": "",
        "person": "Mia"
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
        "description": "",
        "person": "family",
        "short_title": "Pike Place"
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
        "description": "",
        "people": [
         "Jacob",
         "Carina"
        ],
        "short_title": "Date night"
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
        "description": "",
        "person": "family"
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
        "description": "",
        "person": "Jacob",
        "leave_by": "2:45 PM",
        "short_title": "Sea-Tac run"
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
        "description": "",
        "person": "Erik"
       }
      ],
      "mode": null,
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
      "weather": {
       "temp": "46",
       "condition": "Overcast",
       "icon": "cloudy",
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
     },
     {
      "type": "say",
      "greeting": "",
      "text": "",
      "events": [
       {
        "id": "evt24",
        "title": "Walk the dog",
        "time": "7:30 AM",
        "when": "Today · 7:30 AM",
        "start": "2026-11-24T07:30:00-08:00",
        "end": "2026-11-24T07:55:00-08:00",
        "location": "Ravenna Park",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
       },
       {
        "id": "evt25",
        "title": "School drop-off",
        "time": "8:05 AM",
        "when": "Today · 8:05 AM",
        "start": "2026-11-24T08:05:00-08:00",
        "end": "2026-11-24T08:30:00-08:00",
        "location": "Erik and Mia",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
       },
       {
        "id": "evt26",
        "title": "Design review",
        "time": "9:00 AM",
        "when": "Today · 9:00 AM",
        "start": "2026-11-24T09:00:00-08:00",
        "end": "2026-11-24T09:25:00-08:00",
        "location": "video call",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
       },
       {
        "id": "evt27",
        "title": "Plumber visit",
        "time": "10:00 AM",
        "when": "Today · 10:00 AM",
        "start": "2026-11-24T10:00:00-08:00",
        "end": "2026-11-24T10:25:00-08:00",
        "location": "home",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Carina"
       },
       {
        "id": "evt28",
        "title": "Grocery pickup",
        "time": "11:00 AM",
        "when": "Today · 11:00 AM",
        "start": "2026-11-24T11:00:00-08:00",
        "end": "2026-11-24T11:25:00-08:00",
        "location": "PCC",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Carina"
       },
       {
        "id": "evt29",
        "title": "Lunch with Dana",
        "time": "12:15 PM",
        "when": "Today · 12:15 PM",
        "start": "2026-11-24T12:15:00-08:00",
        "end": "2026-11-24T12:40:00-08:00",
        "location": "Ballard",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Carina"
       },
       {
        "id": "evt30",
        "title": "Quarterly planning with the whole product group",
        "time": "1:30 PM",
        "when": "Today · 1:30 PM",
        "start": "2026-11-24T13:30:00-08:00",
        "end": "2026-11-24T13:55:00-08:00",
        "location": "video call",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
       },
       {
        "id": "evt31",
        "title": "Library returns",
        "time": "2:30 PM",
        "when": "Today · 2:30 PM",
        "start": "2026-11-24T14:30:00-08:00",
        "end": "2026-11-24T14:55:00-08:00",
        "location": "Northeast branch",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Mia"
       },
       {
        "id": "evt32",
        "title": "School pickup",
        "time": "3:10 PM",
        "when": "Today · 3:10 PM",
        "start": "2026-11-24T15:10:00-08:00",
        "end": "2026-11-24T15:35:00-08:00",
        "location": "early release",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Carina"
       },
       {
        "id": "evt33",
        "title": "Piano lesson",
        "time": "4:00 PM",
        "when": "Today · 4:00 PM",
        "start": "2026-11-24T16:00:00-08:00",
        "end": "2026-11-24T16:25:00-08:00",
        "location": "Ravenna",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Mia"
       },
       {
        "id": "evt34",
        "title": "Swim practice",
        "time": "5:30 PM",
        "when": "Today · 5:30 PM",
        "start": "2026-11-24T17:30:00-08:00",
        "end": "2026-11-24T17:55:00-08:00",
        "location": "Green Lake pool",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Erik",
        "leave_by": "5:10 PM"
       },
       {
        "id": "evt35",
        "title": "Dinner",
        "time": "6:30 PM",
        "when": "Today · 6:30 PM",
        "start": "2026-11-24T18:30:00-08:00",
        "end": "2026-11-24T18:55:00-08:00",
        "location": "home",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "family"
       },
       {
        "id": "evt36",
        "title": "Parent–teacher conference",
        "time": "7:00 PM",
        "when": "Today · 7:00 PM",
        "start": "2026-11-24T19:00:00-08:00",
        "end": "2026-11-24T19:25:00-08:00",
        "location": "Bryant Elementary",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "family"
       },
       {
        "id": "evt37",
        "title": "Bins to the curb",
        "time": "8:30 PM",
        "when": "Today · 8:30 PM",
        "start": "2026-11-24T20:30:00-08:00",
        "end": "2026-11-24T20:55:00-08:00",
        "location": "",
        "duration": "25m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
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
        "description": "",
        "people": [
         "Erik",
         "Mia"
        ]
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
        "description": "",
        "person": "family",
        "leave_by": "2026-11-25T07:25:00-08:00",
        "short_title": "Ferry"
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
        "description": "",
        "person": "Mia",
        "short_title": "Piano"
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
        "description": "",
        "person": "Carina"
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
        "description": "",
        "kind": "holiday"
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
        "description": "",
        "person": "family",
        "short_title": "Grandma June"
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
        "description": "",
        "person": "Jacob",
        "short_title": "Turkey trot"
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
        "description": "",
        "person": "family",
        "short_title": "Dinner at 3"
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
        "description": "",
        "person": "Erik",
        "short_title": "Soccer"
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
        "description": "",
        "person": "Mia"
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
        "description": "",
        "person": "family",
        "short_title": "Pike Place"
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
        "description": "",
        "people": [
         "Jacob",
         "Carina"
        ],
        "short_title": "Date night"
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
        "description": "",
        "person": "family"
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
        "description": "",
        "person": "Jacob",
        "leave_by": "2:45 PM",
        "short_title": "Sea-Tac run"
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
        "description": "",
        "person": "Erik"
       }
      ],
      "who": "Jacob"
     }
    ]
   },
   {
    "type": "zone",
    "place": "ground",
    "children": [
     {
      "type": "todo_list",
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
      "max": 3
     },
     {
      "type": "forecast_strip",
      "days": [
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
      "count": 5
     }
    ]
   }
  ]
 },
 "data": {
  "weather": {
   "temp": "46",
   "condition": "Overcast",
   "icon": "cloudy",
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
     "id": "evt24",
     "title": "Walk the dog",
     "time": "7:30 AM",
     "when": "Today · 7:30 AM",
     "start": "2026-11-24T07:30:00-08:00",
     "end": "2026-11-24T07:55:00-08:00",
     "location": "Ravenna Park",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Jacob"
    },
    {
     "id": "evt25",
     "title": "School drop-off",
     "time": "8:05 AM",
     "when": "Today · 8:05 AM",
     "start": "2026-11-24T08:05:00-08:00",
     "end": "2026-11-24T08:30:00-08:00",
     "location": "Erik and Mia",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Jacob"
    },
    {
     "id": "evt26",
     "title": "Design review",
     "time": "9:00 AM",
     "when": "Today · 9:00 AM",
     "start": "2026-11-24T09:00:00-08:00",
     "end": "2026-11-24T09:25:00-08:00",
     "location": "video call",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Jacob"
    },
    {
     "id": "evt27",
     "title": "Plumber visit",
     "time": "10:00 AM",
     "when": "Today · 10:00 AM",
     "start": "2026-11-24T10:00:00-08:00",
     "end": "2026-11-24T10:25:00-08:00",
     "location": "home",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Carina"
    },
    {
     "id": "evt28",
     "title": "Grocery pickup",
     "time": "11:00 AM",
     "when": "Today · 11:00 AM",
     "start": "2026-11-24T11:00:00-08:00",
     "end": "2026-11-24T11:25:00-08:00",
     "location": "PCC",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Carina"
    },
    {
     "id": "evt29",
     "title": "Lunch with Dana",
     "time": "12:15 PM",
     "when": "Today · 12:15 PM",
     "start": "2026-11-24T12:15:00-08:00",
     "end": "2026-11-24T12:40:00-08:00",
     "location": "Ballard",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Carina"
    },
    {
     "id": "evt30",
     "title": "Quarterly planning with the whole product group",
     "time": "1:30 PM",
     "when": "Today · 1:30 PM",
     "start": "2026-11-24T13:30:00-08:00",
     "end": "2026-11-24T13:55:00-08:00",
     "location": "video call",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Jacob"
    },
    {
     "id": "evt31",
     "title": "Library returns",
     "time": "2:30 PM",
     "when": "Today · 2:30 PM",
     "start": "2026-11-24T14:30:00-08:00",
     "end": "2026-11-24T14:55:00-08:00",
     "location": "Northeast branch",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Mia"
    },
    {
     "id": "evt32",
     "title": "School pickup",
     "time": "3:10 PM",
     "when": "Today · 3:10 PM",
     "start": "2026-11-24T15:10:00-08:00",
     "end": "2026-11-24T15:35:00-08:00",
     "location": "early release",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Carina"
    },
    {
     "id": "evt33",
     "title": "Piano lesson",
     "time": "4:00 PM",
     "when": "Today · 4:00 PM",
     "start": "2026-11-24T16:00:00-08:00",
     "end": "2026-11-24T16:25:00-08:00",
     "location": "Ravenna",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Mia"
    },
    {
     "id": "evt34",
     "title": "Swim practice",
     "time": "5:30 PM",
     "when": "Today · 5:30 PM",
     "start": "2026-11-24T17:30:00-08:00",
     "end": "2026-11-24T17:55:00-08:00",
     "location": "Green Lake pool",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Erik",
     "leave_by": "5:10 PM"
    },
    {
     "id": "evt35",
     "title": "Dinner",
     "time": "6:30 PM",
     "when": "Today · 6:30 PM",
     "start": "2026-11-24T18:30:00-08:00",
     "end": "2026-11-24T18:55:00-08:00",
     "location": "home",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "family"
    },
    {
     "id": "evt36",
     "title": "Parent–teacher conference",
     "time": "7:00 PM",
     "when": "Today · 7:00 PM",
     "start": "2026-11-24T19:00:00-08:00",
     "end": "2026-11-24T19:25:00-08:00",
     "location": "Bryant Elementary",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "family"
    },
    {
     "id": "evt37",
     "title": "Bins to the curb",
     "time": "8:30 PM",
     "when": "Today · 8:30 PM",
     "start": "2026-11-24T20:30:00-08:00",
     "end": "2026-11-24T20:55:00-08:00",
     "location": "",
     "duration": "25m",
     "all_day": false,
     "description": "",
     "person": "Jacob"
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
     "description": "",
     "people": [
      "Erik",
      "Mia"
     ]
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
     "description": "",
     "person": "family",
     "leave_by": "2026-11-25T07:25:00-08:00",
     "short_title": "Ferry"
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
     "description": "",
     "person": "Mia",
     "short_title": "Piano"
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
     "description": "",
     "person": "Carina"
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
     "description": "",
     "kind": "holiday"
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
     "description": "",
     "person": "family",
     "short_title": "Grandma June"
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
     "description": "",
     "person": "Jacob",
     "short_title": "Turkey trot"
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
     "description": "",
     "person": "family",
     "short_title": "Dinner at 3"
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
     "description": "",
     "person": "Erik",
     "short_title": "Soccer"
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
     "description": "",
     "person": "Mia"
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
     "description": "",
     "person": "family",
     "short_title": "Pike Place"
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
     "description": "",
     "people": [
      "Jacob",
      "Carina"
     ],
     "short_title": "Date night"
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
     "description": "",
     "person": "family"
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
     "description": "",
     "person": "Jacob",
     "leave_by": "2:45 PM",
     "short_title": "Sea-Tac run"
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
     "description": "",
     "person": "Erik"
    }
   ],
   "count": 29
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
    }
   ],
   "count": 1
  },
  "brief": {
   "greeting": "",
   "summary": ""
  },
  "clock": {
   "hour": 12,
   "minute": 40,
   "second": 0,
   "display": "12:40",
   "date": "November 24, 2026",
   "date_short": "November 24",
   "day_of_week": "Tuesday"
  }
 },
 "args": {},
 "status": null,
 "presence": null
};
