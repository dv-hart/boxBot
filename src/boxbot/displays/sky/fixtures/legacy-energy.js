/* generated from the real display spec + placeholder data (dev fixture) */
BB.fixture = {
 "v": 1,
 "display": "energy",
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
      "type": "icon",
      "name": "zap",
      "size": "lg",
      "color": "accent"
     },
     {
      "type": "text",
      "content": "Home Energy",
      "size": "title"
     }
    ]
   },
   {
    "type": "columns",
    "ratios": [
     1,
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
      "gap": 16,
      "children": [
       {
        "type": "text",
        "content": "Right Now",
        "size": "heading"
       },
       {
        "type": "metric",
        "value": "3.1 kW",
        "label": "Solar",
        "icon": "sun"
       },
       {
        "type": "metric",
        "value": "1.8 kW",
        "label": "Usage",
        "icon": "home"
       },
       {
        "type": "metric",
        "value": "1.3 kW",
        "label": "Net Export",
        "icon": "arrow-up-right"
       },
       {
        "type": "spacer"
       },
       {
        "type": "progress",
        "value": "0.64",
        "label": "Battery 64%",
        "color": "success"
       }
      ]
     },
     {
      "type": "column",
      "gap": 12,
      "children": [
       {
        "type": "text",
        "content": "Today",
        "size": "heading"
       },
       {
        "type": "area",
        "height": 280,
        "series": [
         {
          "data": [
           0,
           0,
           0.2,
           0.9,
           1.8,
           2.7,
           3.2,
           3.1,
           2.4,
           1.2,
           0.3,
           0
          ],
          "color": "success",
          "label": "Solar"
         },
         {
          "data": [
           0.6,
           0.5,
           0.9,
           1.6,
           1.2,
           1.0,
           1.4,
           1.8,
           2.2,
           2.6,
           1.9,
           1.1
          ],
          "color": "warning",
          "label": "Usage"
         }
        ],
        "x_labels": [
         "6a",
         "",
         "",
         "",
         "",
         "12p",
         "",
         "",
         "",
         "",
         "",
         "6p"
        ],
        "show_legend": true,
        "fill_opacity": 0.1,
        "chart_type": "area"
       }
      ]
     }
    ]
   }
  ]
 },
 "data": {
  "energy": {
   "solar_kw": "3.1",
   "usage_kw": "1.8",
   "net_kw": "1.3",
   "battery_pct": "0.64",
   "solar_hourly": [
    0,
    0,
    0.2,
    0.9,
    1.8,
    2.7,
    3.2,
    3.1,
    2.4,
    1.2,
    0.3,
    0
   ],
   "usage_hourly": [
    0.6,
    0.5,
    0.9,
    1.6,
    1.2,
    1.0,
    1.4,
    1.8,
    2.2,
    2.6,
    1.9,
    1.1
   ],
   "hours": [
    "6a",
    "",
    "",
    "",
    "",
    "12p",
    "",
    "",
    "",
    "",
    "",
    "6p"
   ]
  }
 },
 "args": {},
 "status": null,
 "presence": null
};
