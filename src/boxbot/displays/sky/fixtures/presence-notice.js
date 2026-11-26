/* dev fixture: presence dock / status pill over the legacy `home_metrics` display */
BB.fixture = {
 "v": 1,
 "display": "home_metrics",
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
  "align": "center",
  "padding": [
   24,
   32,
   24,
   32
  ],
  "children": [
   {
    "type": "row",
    "align": "start",
    "gap": 0,
    "children": [
     {
      "type": "text",
      "content": "Home",
      "size": "heading",
      "weight": "bold"
     },
     {
      "type": "spacer"
     },
     {
      "type": "clock",
      "show_date": false,
      "size": "md",
      "show_seconds": false
     }
    ]
   },
   {
    "type": "columns",
    "ratios": [
     1,
     1,
     1,
     1
    ],
    "gap": 16,
    "children": [
     {
      "type": "card",
      "padding": 18,
      "children": [
       {
        "type": "column",
        "gap": 8,
        "children": [
         {
          "type": "row",
          "gap": 8,
          "align": "start",
          "children": [
           {
            "type": "icon",
            "name": "cloud-sun",
            "size": "sm",
            "color": "accent"
           },
           {
            "type": "text",
            "content": "OUTSIDE",
            "size": "small",
            "color": "muted",
            "weight": "semibold"
           }
          ]
         },
         {
          "type": "text",
          "content": "72°",
          "size": "title",
          "weight": "bold"
         },
         {
          "type": "text",
          "content": "Partly Cloudy",
          "size": "caption",
          "color": "muted",
          "max_lines": 1
         }
        ]
       }
      ]
     },
     {
      "type": "card",
      "padding": 18,
      "children": [
       {
        "type": "column",
        "gap": 8,
        "children": [
         {
          "type": "row",
          "gap": 8,
          "align": "start",
          "children": [
           {
            "type": "icon",
            "name": "home",
            "size": "sm",
            "color": "muted"
           },
           {
            "type": "text",
            "content": "INSIDE",
            "size": "small",
            "color": "muted",
            "weight": "semibold"
           }
          ]
         },
         {
          "type": "text",
          "content": "68°",
          "size": "title",
          "weight": "bold"
         },
         {
          "type": "text",
          "content": "Living room",
          "size": "caption",
          "color": "muted"
         }
        ]
       }
      ]
     },
     {
      "type": "card",
      "padding": 18,
      "children": [
       {
        "type": "column",
        "gap": 8,
        "children": [
         {
          "type": "row",
          "gap": 8,
          "align": "start",
          "children": [
           {
            "type": "icon",
            "name": "wifi",
            "size": "sm",
            "color": "muted"
           },
           {
            "type": "text",
            "content": "NETWORK",
            "size": "small",
            "color": "muted",
            "weight": "semibold"
           }
          ]
         },
         {
          "type": "text",
          "content": "112",
          "size": "title",
          "weight": "bold"
         },
         {
          "type": "text",
          "content": "ms · 1 Gbps",
          "size": "caption",
          "color": "muted"
         }
        ]
       }
      ]
     },
     {
      "type": "card",
      "padding": 18,
      "children": [
       {
        "type": "column",
        "gap": 8,
        "children": [
         {
          "type": "row",
          "gap": 8,
          "align": "start",
          "children": [
           {
            "type": "icon",
            "name": "battery-full",
            "size": "sm",
            "color": "success"
           },
           {
            "type": "text",
            "content": "POWER",
            "size": "small",
            "color": "muted",
            "weight": "semibold"
           }
          ]
         },
         {
          "type": "text",
          "content": "87%",
          "size": "title",
          "weight": "bold"
         },
         {
          "type": "text",
          "content": "UPS · on mains",
          "size": "caption",
          "color": "muted"
         }
        ]
       }
      ]
     }
    ]
   },
   {
    "type": "columns",
    "ratios": [
     3,
     2
    ],
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
          "type": "row",
          "align": "start",
          "gap": 0,
          "children": [
           {
            "type": "text",
            "content": "POWER DRAW",
            "size": "small",
            "color": "muted",
            "weight": "semibold"
           },
           {
            "type": "spacer"
           },
           {
            "type": "text",
            "content": "7 days · 38W avg",
            "size": "small",
            "color": "dim"
           }
          ]
         },
         {
          "type": "area",
          "height": 200,
          "data": [
           32,
           28,
           35,
           41,
           38,
           44,
           40
          ],
          "color": "secondary",
          "x_labels": [
           "Fri",
           "Sat",
           "Sun",
           "Mon",
           "Tue",
           "Wed",
           "Thu"
          ],
          "fill_opacity": 0.09,
          "show_dots": true,
          "chart_type": "area"
         }
        ]
       }
      ]
     },
     {
      "type": "card",
      "padding": 20,
      "children": [
       {
        "type": "column",
        "gap": 14,
        "children": [
         {
          "type": "text",
          "content": "SYSTEM",
          "size": "small",
          "color": "muted",
          "weight": "semibold"
         },
         {
          "type": "columns",
          "ratios": [
           2,
           3
          ],
          "gap": 12,
          "children": [
           {
            "type": "text",
            "content": "CPU",
            "color": "muted",
            "size": "body"
           },
           {
            "type": "text",
            "content": "23%",
            "weight": "medium",
            "size": "body"
           }
          ]
         },
         {
          "type": "columns",
          "ratios": [
           2,
           3
          ],
          "gap": 12,
          "children": [
           {
            "type": "text",
            "content": "SoC · Hailo",
            "color": "muted",
            "size": "body"
           },
           {
            "type": "text",
            "content": "51° · 44°",
            "weight": "medium",
            "size": "body"
           }
          ]
         },
         {
          "type": "columns",
          "ratios": [
           2,
           3
          ],
          "gap": 12,
          "children": [
           {
            "type": "text",
            "content": "Memory",
            "color": "muted",
            "size": "body"
           },
           {
            "type": "text",
            "content": "3.2 / 8.0 GB",
            "weight": "medium",
            "size": "body"
           }
          ]
         },
         {
          "type": "columns",
          "ratios": [
           2,
           3
          ],
          "gap": 12,
          "children": [
           {
            "type": "text",
            "content": "Uptime",
            "color": "muted",
            "size": "body"
           },
           {
            "type": "text",
            "content": "3d 14h",
            "weight": "medium",
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
  }
 },
 "args": {},
 "status": null,
 "presence": {
  "state": "notice",
  "who": "Erik",
  "label": "Reminder",
  "detail": "5:00 PM",
  "text": "Leave in 10 minutes for swim practice"
 }
};
