/* generated from the real display spec + placeholder data (dev fixture) */
BB.fixture = {
 "v": 1,
 "display": "dashboard",
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
  "children": [
   {
    "type": "row",
    "gap": 12,
    "align": "center",
    "padding": [
     16,
     24
    ],
    "children": [
     {
      "type": "emoji",
      "name": "sunrise"
     },
     {
      "type": "text",
      "content": "Good Morning",
      "size": "title"
     },
     {
      "type": "spacer"
     },
     {
      "type": "clock",
      "show_date": false,
      "size": "md",
      "format": "12h"
     }
    ]
   },
   {
    "type": "columns",
    "ratios": [
     2,
     1
    ],
    "gap": 24,
    "padding": [
     0,
     24,
     24,
     24
    ],
    "children": [
     {
      "type": "column",
      "gap": 20,
      "children": [
       {
        "type": "card",
        "padding": 16,
        "children": [
         {
          "type": "clock",
          "format": "24h",
          "show_seconds": true,
          "show_date": true,
          "size": "lg"
         }
        ]
       },
       {
        "type": "weather_widget",
        "data_source": "weather"
       }
      ]
     },
     {
      "type": "column",
      "children": [
       {
        "type": "text",
        "content": "Today",
        "size": "heading"
       },
       {
        "type": "spacer",
        "size": 8
       },
       {
        "type": "divider"
       },
       {
        "type": "repeat",
        "source": [
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
        "max": 8,
        "children": [
         {
          "type": "row",
          "gap": 12,
          "padding": [
           10,
           0
          ],
          "children": [
           {
            "type": "text",
            "content": "9:00 AM",
            "size": "caption",
            "color": "muted",
            "min_width": 50
           },
           {
            "type": "divider",
            "color": "accent",
            "orientation": "vertical"
           },
           {
            "type": "column",
            "children": [
             {
              "type": "text",
              "content": "Team Standup",
              "max_lines": 1,
              "size": "body"
             },
             {
              "type": "text",
              "content": "30m",
              "size": "small",
              "color": "muted"
             }
            ]
           }
          ]
         },
         {
          "type": "row",
          "gap": 12,
          "padding": [
           10,
           0
          ],
          "children": [
           {
            "type": "text",
            "content": "11:00 AM",
            "size": "caption",
            "color": "muted",
            "min_width": 50
           },
           {
            "type": "divider",
            "color": "accent",
            "orientation": "vertical"
           },
           {
            "type": "column",
            "children": [
             {
              "type": "text",
              "content": "Design Review",
              "max_lines": 1,
              "size": "body"
             },
             {
              "type": "text",
              "content": "1h",
              "size": "small",
              "color": "muted"
             }
            ]
           }
          ]
         },
         {
          "type": "row",
          "gap": 12,
          "padding": [
           10,
           0
          ],
          "children": [
           {
            "type": "text",
            "content": "2:00 PM",
            "size": "caption",
            "color": "muted",
            "min_width": 50
           },
           {
            "type": "divider",
            "color": "accent",
            "orientation": "vertical"
           },
           {
            "type": "column",
            "children": [
             {
              "type": "text",
              "content": "Dentist Appointment",
              "max_lines": 1,
              "size": "body"
             },
             {
              "type": "text",
              "content": "1h",
              "size": "small",
              "color": "muted"
             }
            ]
           }
          ]
         }
        ]
       }
      ]
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
