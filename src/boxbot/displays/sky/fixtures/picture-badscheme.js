/* picture · non-file sources are refused → empty state */
BB.fixture = {
  "v": 1,
  "display": "picture",
  "people": [
    "Jacob",
    "Carina",
    "Erik",
    "Mia"
  ],
  "now": "2026-11-24T16:12:00-08:00",
  "atmosphere": {
    "phase": "golden",
    "weather": "partly",
    "time": "16:12",
    "dim": false
  },
  "scene": {
    "horizon": "high"
  },
  "layout": {
    "type": "picture",
    "photos": [
      {
        "id": "y",
        "src": "https://example.com/a.jpg",
        "title": "Remote"
      },
      {
        "id": "z",
        "src": "javascript:alert(1)"
      }
    ],
    "layout": "auto",
    "pair": false,
    "reason": null,
    "position": null,
    "ui": "full",
    "minimal_after_s": 6,
    "temp": 48,
    "condition": "Sunbreaks",
    "icon": "cloud-sun"
  },
  "data": {
    "photos": {
      "items": [
        {
          "id": "y",
          "src": "https://example.com/a.jpg",
          "title": "Remote"
        },
        {
          "id": "z",
          "src": "javascript:alert(1)"
        }
      ],
      "count": 2
    },
    "weather": {
      "temp": 48,
      "condition": "Sunbreaks",
      "icon": "cloud-sun"
    }
  },
  "args": {
    "image_ids": [
      "y",
      "z"
    ]
  },
  "status": null,
  "presence": null
};
