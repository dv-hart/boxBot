/* dev fixture: presence dock / status pill over the legacy `morning_brief` display */
BB.fixture = {
 "v": 1,
 "display": "morning_brief",
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
  "gap": 20,
  "align": "center",
  "padding": [
   28,
   36,
   28,
   36
  ],
  "children": [
   {
    "type": "row",
    "align": "start",
    "gap": 0,
    "children": [
     {
      "type": "text",
      "content": "Good morning, Jacob",
      "size": "heading",
      "weight": "bold"
     },
     {
      "type": "spacer"
     },
     {
      "type": "column",
      "gap": 2,
      "item_align": "end",
      "children": [
       {
        "type": "clock",
        "show_date": false,
        "size": "md",
        "show_seconds": false
       },
       {
        "type": "text",
        "content": "Tuesday, November 24",
        "size": "caption",
        "color": "muted",
        "align": "right"
       }
      ]
     }
    ]
   },
   {
    "type": "columns",
    "ratios": [
     5,
     4
    ],
    "gap": 20,
    "grow": true,
    "children": [
     {
      "type": "card",
      "padding": 24,
      "children": [
       {
        "type": "column",
        "gap": 16,
        "grow": true,
        "children": [
         {
          "type": "text",
          "content": "WEATHER",
          "size": "small",
          "color": "muted",
          "weight": "semibold"
         },
         {
          "type": "spacer"
         },
         {
          "type": "row",
          "gap": 18,
          "align": "start",
          "children": [
           {
            "type": "icon",
            "name": "cloud-sun",
            "size": "xl",
            "color": "accent"
           },
           {
            "type": "column",
            "gap": 2,
            "children": [
             {
              "type": "text",
              "content": "72°",
              "size": "title",
              "weight": "bold"
             },
             {
              "type": "text",
              "content": "Partly Cloudy",
              "color": "muted",
              "size": "body"
             }
            ]
           }
          ]
         },
         {
          "type": "spacer"
         },
         {
          "type": "row",
          "gap": 18,
          "align": "spread",
          "children": [
           {
            "type": "column",
            "gap": 6,
            "item_align": "center",
            "children": [
             {
              "type": "text",
              "content": "Mon",
              "size": "caption",
              "color": "muted"
             },
             {
              "type": "icon",
              "name": "sun",
              "size": "sm",
              "color": "muted"
             },
             {
              "type": "text",
              "content": "75°",
              "size": "caption"
             }
            ]
           },
           {
            "type": "column",
            "gap": 6,
            "item_align": "center",
            "children": [
             {
              "type": "text",
              "content": "Tue",
              "size": "caption",
              "color": "muted"
             },
             {
              "type": "icon",
              "name": "cloud",
              "size": "sm",
              "color": "muted"
             },
             {
              "type": "text",
              "content": "68°",
              "size": "caption"
             }
            ]
           },
           {
            "type": "column",
            "gap": 6,
            "item_align": "center",
            "children": [
             {
              "type": "text",
              "content": "Wed",
              "size": "caption",
              "color": "muted"
             },
             {
              "type": "icon",
              "name": "cloud-rain",
              "size": "sm",
              "color": "muted"
             },
             {
              "type": "text",
              "content": "62°",
              "size": "caption"
             }
            ]
           },
           {
            "type": "column",
            "gap": 6,
            "item_align": "center",
            "children": [
             {
              "type": "text",
              "content": "Thu",
              "size": "caption",
              "color": "muted"
             },
             {
              "type": "icon",
              "name": "sun",
              "size": "sm",
              "color": "muted"
             },
             {
              "type": "text",
              "content": "70°",
              "size": "caption"
             }
            ]
           },
           {
            "type": "column",
            "gap": 6,
            "item_align": "center",
            "children": [
             {
              "type": "text",
              "content": "Fri",
              "size": "caption",
              "color": "muted"
             },
             {
              "type": "icon",
              "name": "cloud-sun",
              "size": "sm",
              "color": "muted"
             },
             {
              "type": "text",
              "content": "73°",
              "size": "caption"
             }
            ]
           }
          ]
         }
        ]
       }
      ]
     },
     {
      "type": "column",
      "gap": 16,
      "children": [
       {
        "type": "card",
        "padding": 20,
        "children": [
         {
          "type": "column",
          "gap": 12,
          "children": [
           {
            "type": "text",
            "content": "NEXT UP",
            "size": "small",
            "color": "muted",
            "weight": "semibold"
           },
           {
            "type": "row",
            "gap": 14,
            "valign": "top",
            "align": "start",
            "children": [
             {
              "type": "text",
              "content": "9:00 AM",
              "size": "subtitle",
              "color": "accent",
              "weight": "bold"
             },
             {
              "type": "column",
              "gap": 2,
              "children": [
               {
                "type": "text",
                "content": "Team Standup",
                "weight": "medium",
                "size": "body"
               },
               {
                "type": "text",
                "content": "Zoom",
                "size": "caption",
                "color": "muted"
               }
              ]
             }
            ]
           }
          ]
         }
        ]
       },
       {
        "type": "card",
        "padding": 20,
        "grow": true,
        "children": [
         {
          "type": "column",
          "gap": 12,
          "children": [
           {
            "type": "row",
            "align": "start",
            "gap": 0,
            "children": [
             {
              "type": "text",
              "content": "TO-DO",
              "size": "small",
              "color": "muted",
              "weight": "semibold"
             },
             {
              "type": "spacer"
             },
             {
              "type": "badge",
              "text": 3,
              "color": "accent"
             }
            ]
           },
           {
            "type": "repeat",
            "source": [
             {
              "description": "Buy groceries",
              "due_date": "Today",
              "status": "pending"
             },
             {
              "description": "Review budget spreadsheet",
              "due_date": "Tomorrow",
              "status": "pending"
             },
             {
              "description": "Call plumber about leak",
              "due_date": null,
              "status": "pending"
             }
            ],
            "max": 3,
            "children": [
             {
              "type": "row",
              "gap": 12,
              "align": "start",
              "children": [
               {
                "type": "icon",
                "name": "circle",
                "size": "sm",
                "color": "dim"
               },
               {
                "type": "text",
                "content": "Buy groceries",
                "max_lines": 1,
                "size": "body"
               }
              ]
             },
             {
              "type": "row",
              "gap": 12,
              "align": "start",
              "children": [
               {
                "type": "icon",
                "name": "circle",
                "size": "sm",
                "color": "dim"
               },
               {
                "type": "text",
                "content": "Review budget spreadsheet",
                "max_lines": 1,
                "size": "body"
               }
              ]
             },
             {
              "type": "row",
              "gap": 12,
              "align": "start",
              "children": [
               {
                "type": "icon",
                "name": "circle",
                "size": "sm",
                "color": "dim"
               },
               {
                "type": "text",
                "content": "Call plumber about leak",
                "max_lines": 1,
                "size": "body"
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
  },
  "tasks": {
   "items": [
    {
     "description": "Buy groceries",
     "due_date": "Today",
     "status": "pending"
    },
    {
     "description": "Review budget spreadsheet",
     "due_date": "Tomorrow",
     "status": "pending"
    },
    {
     "description": "Call plumber about leak",
     "due_date": null,
     "status": "pending"
    }
   ],
   "count": 3
  },
  "people": {
   "present": [
    {
     "name": "Jacob",
     "since": "2 min ago"
    }
   ],
   "count": 1
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
 "presence": {
  "state": "speaking",
  "text": "Thursday at 6:30, at Benaroya Hall. Doors open at six."
 }
};
