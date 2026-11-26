/* dev fixture: presence dock / status pill over the legacy `weather_simple` display */
BB.fixture = {
 "v": 1,
 "display": "weather_simple",
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
  "gap": 10,
  "align": "center",
  "item_align": "center",
  "padding": [
   40,
   40,
   40,
   40
  ],
  "children": [
   {
    "type": "row",
    "gap": 20,
    "align": "center",
    "children": [
     {
      "type": "icon",
      "name": "cloud-sun",
      "size": "xl",
      "color": "accent"
     },
     {
      "type": "text",
      "content": "72°",
      "size": "title",
      "weight": "bold"
     }
    ]
   },
   {
    "type": "text",
    "content": "Partly Cloudy",
    "size": "subtitle",
    "color": "muted",
    "align": "center"
   },
   {
    "type": "spacer",
    "size": 14
   },
   {
    "type": "row",
    "gap": 12,
    "align": "center",
    "children": [
     {
      "type": "text",
      "content": "Humidity 65%",
      "size": "caption",
      "color": "muted"
     },
     {
      "type": "text",
      "content": "·",
      "size": "caption",
      "color": "dim"
     },
     {
      "type": "text",
      "content": "Wind 12 mph",
      "size": "caption",
      "color": "muted"
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
 "status": "Searching the web for ferry times…",
 "presence": null
};
