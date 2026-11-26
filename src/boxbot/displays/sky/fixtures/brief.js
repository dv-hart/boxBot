/* brief — specs/brief.json with its bindings resolved against the sample data below (Tue 24 Nov 2026, Seattle).
   Morning is the design's dawn-fog brief (7:12). Other hours: &now=2026-11-24T16:02:00-08:00&phase=golden&weather=partly (the sentence is static, so it stays the morning one).
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
  "sun": {
   "next": "sunrise",
   "at": "7:28 AM"
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
        "id": "evt03",
        "title": "School drop-off",
        "time": "8:05 AM",
        "when": "Today · 8:05 AM",
        "start": "2026-11-24T08:05:00-08:00",
        "end": "2026-11-24T08:25:00-08:00",
        "location": "Erik and Mia",
        "duration": "20m",
        "all_day": false,
        "description": "",
        "person": "Jacob",
        "leave_by": "7:50 AM",
        "short_title": "Drop-off"
       },
       {
        "id": "evt04",
        "title": "Design review",
        "time": "9:00 AM",
        "when": "Today · 9:00 AM",
        "start": "2026-11-24T09:00:00-08:00",
        "end": "2026-11-24T09:45:00-08:00",
        "location": "video call",
        "duration": "45m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
       },
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
        "description": "",
        "person": "Carina",
        "leave_by": "11:50 AM",
        "short_title": "Lunch, Dana"
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
        "description": "",
        "person": "Carina",
        "leave_by": "2:55 PM",
        "short_title": "Pickup"
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
        "description": "",
        "person": "Erik",
        "leave_by": "5:10 PM"
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
        "description": "",
        "person": "family",
        "leave_by": "6:40 PM",
        "short_title": "Conferences"
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
       "temp": "41",
       "condition": "Fog",
       "icon": "cloud-fog",
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
      "greeting": "Good morning, Jacob",
      "text": "Six things today. Drop-off comes first at 8:05, and the fog should lift by ten.",
      "events": [
       {
        "id": "evt03",
        "title": "School drop-off",
        "time": "8:05 AM",
        "when": "Today · 8:05 AM",
        "start": "2026-11-24T08:05:00-08:00",
        "end": "2026-11-24T08:25:00-08:00",
        "location": "Erik and Mia",
        "duration": "20m",
        "all_day": false,
        "description": "",
        "person": "Jacob",
        "leave_by": "7:50 AM",
        "short_title": "Drop-off"
       },
       {
        "id": "evt04",
        "title": "Design review",
        "time": "9:00 AM",
        "when": "Today · 9:00 AM",
        "start": "2026-11-24T09:00:00-08:00",
        "end": "2026-11-24T09:45:00-08:00",
        "location": "video call",
        "duration": "45m",
        "all_day": false,
        "description": "",
        "person": "Jacob"
       },
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
        "description": "",
        "person": "Carina",
        "leave_by": "11:50 AM",
        "short_title": "Lunch, Dana"
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
        "description": "",
        "person": "Carina",
        "leave_by": "2:55 PM",
        "short_title": "Pickup"
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
        "description": "",
        "person": "Erik",
        "leave_by": "5:10 PM"
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
        "description": "",
        "person": "family",
        "leave_by": "6:40 PM",
        "short_title": "Conferences"
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
   "temp": "41",
   "condition": "Fog",
   "icon": "cloud-fog",
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
     "id": "evt03",
     "title": "School drop-off",
     "time": "8:05 AM",
     "when": "Today · 8:05 AM",
     "start": "2026-11-24T08:05:00-08:00",
     "end": "2026-11-24T08:25:00-08:00",
     "location": "Erik and Mia",
     "duration": "20m",
     "all_day": false,
     "description": "",
     "person": "Jacob",
     "leave_by": "7:50 AM",
     "short_title": "Drop-off"
    },
    {
     "id": "evt04",
     "title": "Design review",
     "time": "9:00 AM",
     "when": "Today · 9:00 AM",
     "start": "2026-11-24T09:00:00-08:00",
     "end": "2026-11-24T09:45:00-08:00",
     "location": "video call",
     "duration": "45m",
     "all_day": false,
     "description": "",
     "person": "Jacob"
    },
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
     "description": "",
     "person": "Carina",
     "leave_by": "11:50 AM",
     "short_title": "Lunch, Dana"
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
     "description": "",
     "person": "Carina",
     "leave_by": "2:55 PM",
     "short_title": "Pickup"
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
     "description": "",
     "person": "Erik",
     "leave_by": "5:10 PM"
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
     "description": "",
     "person": "family",
     "leave_by": "6:40 PM",
     "short_title": "Conferences"
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
   "count": 21
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
   "greeting": "Good morning, Jacob",
   "summary": "Six things today. Drop-off comes first at 8:05, and the fog should lift by ten."
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
