/* generated from the real display spec + placeholder data (dev fixture) */
BB.fixture = {
 "v": 1,
 "display": "now_listening",
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
  "padding": [
   36,
   48,
   40,
   48
  ],
  "gap": 0,
  "children": [
   {
    "type": "row",
    "align": "start",
    "gap": 0,
    "children": [
     {
      "type": "row",
      "gap": 10,
      "align": "start",
      "children": [
       {
        "type": "icon",
        "name": "mic",
        "size": "sm",
        "color": "accent"
       },
       {
        "type": "text",
        "content": "listening",
        "size": "caption",
        "color": "accent",
        "weight": "semibold"
       }
      ]
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
    "type": "spacer"
   },
   {
    "type": "column",
    "gap": 10,
    "children": [
     {
      "type": "text",
      "content": "Jacob",
      "size": "title",
      "weight": "bold",
      "align": "center"
     },
     {
      "type": "text",
      "content": "present · 2 min ago",
      "color": "muted",
      "align": "center",
      "size": "body"
     }
    ]
   },
   {
    "type": "spacer"
   },
   {
    "type": "divider",
    "color": "surface_alt"
   },
   {
    "type": "spacer",
    "size": 24
   },
   {
    "type": "row",
    "align": "spread",
    "children": [
     {
      "type": "column",
      "gap": 6,
      "children": [
       {
        "type": "text",
        "content": "LAST HEARD",
        "size": "small",
        "color": "muted",
        "weight": "semibold"
       },
       {
        "type": "text",
        "content": "just now",
        "size": "body"
       }
      ]
     },
     {
      "type": "column",
      "gap": 6,
      "children": [
       {
        "type": "text",
        "content": "IN THE ROOM",
        "size": "small",
        "color": "muted",
        "weight": "semibold",
        "align": "center"
       },
       {
        "type": "row",
        "gap": 8,
        "align": "center",
        "children": [
         {
          "type": "icon",
          "name": "users",
          "size": "sm",
          "color": "muted"
         },
         {
          "type": "text",
          "content": 1,
          "weight": "medium",
          "size": "body"
         }
        ]
       }
      ]
     },
     {
      "type": "column",
      "gap": 6,
      "children": [
       {
        "type": "text",
        "content": "NEXT WAKE",
        "size": "small",
        "color": "muted",
        "weight": "semibold",
        "align": "right"
       },
       {
        "type": "text",
        "content": "7:00 AM",
        "align": "right",
        "size": "body"
       }
      ]
     }
    ]
   }
  ]
 },
 "data": {
  "agent_status": {
   "state": "listening",
   "last_active": "just now",
   "next_wake": "7:00 AM"
  },
  "people": {
   "present": [
    {
     "name": "Jacob",
     "since": "2 min ago"
    }
   ],
   "count": 1
  }
 },
 "args": {},
 "status": null,
 "presence": null
};
