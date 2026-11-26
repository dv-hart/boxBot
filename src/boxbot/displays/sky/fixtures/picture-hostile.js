/* picture · markup and script in every string field: must render as literal text */
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
        "id": "x",
        "src": "fixtures/img/alki-43.jpg",
        "thumb": "javascript:alert(1)",
        "width": 1024,
        "height": 768,
        "title": "<b onclick=\"x()\">Bold</b> & \"quoted\" 'title'",
        "description": "<img src=x onerror=\"document.body.innerHTML=1\">",
        "people": [
          {
            "name": "<i>Mia</i>",
            "box": [
              0.326,
              0.672,
              0.05,
              0.058
            ]
          },
          {
            "name": "Erik\" onmouseover=\"x",
            "box": "nope"
          }
        ],
        "taken": "</span><script>alert(1)</script>",
        "source": "x\"><script>alert(1)</script>"
      }
    ],
    "layout": "auto",
    "pair": false,
    "reason": "</div><script>alert(1)</script><img src=x onerror=\"document.body.innerHTML=1\">",
    "position": {
      "index": "2",
      "count": "4"
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
          "id": "x",
          "src": "fixtures/img/alki-43.jpg",
          "thumb": "javascript:alert(1)",
          "width": 1024,
          "height": 768,
          "title": "<b onclick=\"x()\">Bold</b> & \"quoted\" 'title'",
          "description": "<img src=x onerror=\"document.body.innerHTML=1\">",
          "people": [
            {
              "name": "<i>Mia</i>",
              "box": [
                0.326,
                0.672,
                0.05,
                0.058
              ]
            },
            {
              "name": "Erik\" onmouseover=\"x",
              "box": "nope"
            }
          ],
          "taken": "</span><script>alert(1)</script>",
          "source": "x\"><script>alert(1)</script>"
        }
      ],
      "count": 1
    },
    "weather": {
      "temp": 48,
      "condition": "Sunbreaks",
      "icon": "cloud-sun"
    }
  },
  "args": {
    "image_ids": [
      "x"
    ]
  },
  "status": null,
  "presence": null
};
