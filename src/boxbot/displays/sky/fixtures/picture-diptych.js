/* picture · pair:true with two portraits → diptych, slim sky column */
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
        "id": "ph_needle",
        "src": "fixtures/img/needle-34.jpg",
        "thumb": "fixtures/img/needle-34-thumb.jpg",
        "width": 600,
        "height": 800,
        "title": "Under the Needle",
        "description": "Two children seen from behind stand on a rain-wet plaza at dusk looking up at the Space Needle, one pointing at the lit observation deck.",
        "people": [
          {
            "name": "Erik",
            "box": [
              0.25,
              0.735,
              0.085,
              0.064
            ]
          },
          {
            "name": "Mia",
            "box": [
              0.4,
              0.775,
              0.07,
              0.05
            ]
          }
        ],
        "taken": "2026-10-04T15:40:00-07:00",
        "source": "Jacob’s phone"
      },
      {
        "id": "ph_fountain",
        "src": "fixtures/img/fountain-916.jpg",
        "thumb": "fixtures/img/fountain-916-thumb.jpg",
        "width": 450,
        "height": 800,
        "title": "The fountain at Seattle Center",
        "description": "A young girl in a pink jacket raises both arms in front of a large domed fountain spraying arcs of water on a bright summer day, with other visitors and trees in the background.",
        "people": [
          {
            "name": "Mia",
            "box": [
              0.33,
              0.7,
              0.12,
              0.07
            ]
          }
        ],
        "taken": "2026-08-15T13:05:00-07:00",
        "source": "Carina’s phone"
      }
    ],
    "layout": "auto",
    "pair": true,
    "reason": "On this day last month",
    "position": {
      "index": 7,
      "count": 12
    },
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
          "id": "ph_needle",
          "src": "fixtures/img/needle-34.jpg",
          "thumb": "fixtures/img/needle-34-thumb.jpg",
          "width": 600,
          "height": 800,
          "title": "Under the Needle",
          "description": "Two children seen from behind stand on a rain-wet plaza at dusk looking up at the Space Needle, one pointing at the lit observation deck.",
          "people": [
            {
              "name": "Erik",
              "box": [
                0.25,
                0.735,
                0.085,
                0.064
              ]
            },
            {
              "name": "Mia",
              "box": [
                0.4,
                0.775,
                0.07,
                0.05
              ]
            }
          ],
          "taken": "2026-10-04T15:40:00-07:00",
          "source": "Jacob’s phone"
        },
        {
          "id": "ph_fountain",
          "src": "fixtures/img/fountain-916.jpg",
          "thumb": "fixtures/img/fountain-916-thumb.jpg",
          "width": 450,
          "height": 800,
          "title": "The fountain at Seattle Center",
          "description": "A young girl in a pink jacket raises both arms in front of a large domed fountain spraying arcs of water on a bright summer day, with other visitors and trees in the background.",
          "people": [
            {
              "name": "Mia",
              "box": [
                0.33,
                0.7,
                0.12,
                0.07
              ]
            }
          ],
          "taken": "2026-08-15T13:05:00-07:00",
          "source": "Carina’s phone"
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
      "ph_needle",
      "ph_fountain"
    ]
  },
  "status": null,
  "presence": null
};
