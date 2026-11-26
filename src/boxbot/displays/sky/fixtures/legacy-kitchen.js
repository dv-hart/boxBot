/* generated from the real display spec + placeholder data (dev fixture) */
BB.fixture = {
 "v": 1,
 "display": "kitchen",
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
  "gap": 14,
  "padding": [
   24,
   32,
   24,
   32
  ],
  "children": [
   {
    "type": "row",
    "gap": 12,
    "children": [
     {
      "type": "emoji",
      "name": "sunrise",
      "size": "lg"
     },
     {
      "type": "text",
      "content": "Kitchen sink",
      "size": "heading"
     },
     {
      "type": "badge",
      "text": "Live"
     },
     {
      "type": "badge",
      "text": "3 new",
      "color": "success"
     },
     {
      "type": "badge",
      "text": "Late",
      "color": "error"
     },
     {
      "type": "badge",
      "text": "Soon",
      "color": "warning"
     },
     {
      "type": "spacer"
     },
     {
      "type": "countdown",
      "target": "2026-11-24T17:30:00-08:00",
      "label": "Swim practice"
     }
    ]
   },
   {
    "type": "columns",
    "ratios": [
     1,
     1,
     1
    ],
    "gap": 16,
    "grow": true,
    "children": [
     {
      "type": "card",
      "padding": 18,
      "children": [
       {
        "type": "metric",
        "value": "$182.40",
        "label": "AAPL",
        "icon": "trending-up",
        "change": "+2.3%"
       },
       {
        "type": "metric",
        "value": "3.1 kW",
        "label": "Solar",
        "icon": "sun",
        "change": "-0.4"
       },
       {
        "type": "spacer"
       },
       {
        "type": "progress",
        "value": 0.42,
        "label": "Battery 42%"
       },
       {
        "type": "progress",
        "value": 0.72,
        "label": "Disk"
       },
       {
        "type": "progress",
        "value": 0.93,
        "label": "Memory"
       },
       {
        "type": "progress",
        "value": 0.5,
        "color": "accent"
       }
      ]
     },
     {
      "type": "card",
      "padding": 18,
      "children": [
       {
        "type": "list",
        "items": [
         "Buy groceries",
         "Review the budget spreadsheet before Friday's call",
         "Call plumber",
         "Water plants",
         "Renew passport"
        ],
        "max_items": 3
       },
       {
        "type": "divider"
       },
       {
        "type": "list",
        "items": [
         "Preheat oven",
         "Roast 40 min"
        ],
        "style": "number"
       },
       {
        "type": "list",
        "items": [
         "Tickets booked",
         "Bags packed"
        ],
        "style": "check"
       },
       {
        "type": "divider"
       },
       {
        "type": "key_value",
        "data": {
         "Humidity": "65%",
         "Wind": "12 mph",
         "Sunset": "4:24 PM"
        }
       }
      ]
     },
     {
      "type": "column",
      "gap": 12,
      "children": [
       {
        "type": "table",
        "headers": [
         "Route",
         "Departs",
         "Status"
        ],
        "rows": [
         [
          "Bainbridge",
          "4:15",
          "On time"
         ],
         [
          "Bremerton",
          "4:40",
          "Delayed 10 min"
         ],
         [
          "Kingston",
          "5:05",
          "On time"
         ]
        ],
        "striped": true
       },
       {
        "type": "bar",
        "height": 120,
        "data": [
         3,
         5,
         2,
         6,
         4,
         7,
         5
        ],
        "x_labels": [
         "M",
         "T",
         "W",
         "T",
         "F",
         "S",
         "S"
        ],
        "chart_type": "bar"
       },
       {
        "type": "row",
        "gap": 12,
        "valign": "top",
        "children": [
         {
          "type": "image",
          "source": "photo:abc123",
          "radius": 12
         },
         {
          "type": "column",
          "gap": 4,
          "children": [
           {
            "type": "text",
            "content": "A long caption that wraps onto more than one line and is then cut",
            "size": "caption",
            "color": "muted",
            "max_lines": 2
           },
           {
            "type": "row",
            "gap": 10,
            "children": [
             {
              "type": "icon",
              "name": "wifi",
              "size": "sm"
             },
             {
              "type": "icon",
              "name": "home"
             },
             {
              "type": "icon",
              "name": "zap",
              "size": "lg",
              "color": "warning"
             },
             {
              "type": "icon",
              "name": "nonexistent-icon",
              "size": "lg"
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
  }
 },
 "args": {},
 "status": null,
 "presence": null
};
