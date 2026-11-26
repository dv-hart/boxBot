/* inbox · markup and script in sender / role / summary / due / owners: literal text only */
BB.fixture = {
  "v": 1,
  "display": "inbox",
  "people": [
    "Jacob",
    "Carina",
    "Erik",
    "Mia"
  ],
  "now": "2026-11-24T14:41:00-08:00",
  "atmosphere": {
    "phase": "afternoon",
    "weather": "rain",
    "time": "14:41",
    "dim": false
  },
  "scene": {
    "horizon": 300,
    "scale": 0.62
  },
  "layout": {
    "type": "inbox",
    "mode": "triage",
    "focus": "1",
    "privacy": "open",
    "count": 3,
    "owners": {
      "<b>carina</b>": 2,
      "jacob": "x"
    },
    "threads": [
      {
        "sender": "<img src=x onerror=\"document.body.innerHTML=1\">",
        "role": "<u>role</u> & \"co\"",
        "for": [
          "carina",
          "<i>bob</i>"
        ],
        "summary": "</p></div><script>alert(1)</script> summary",
        "short": "Conference Tuesday: 3:40 or 4:20?",
        "due": "<b>soon</b>",
        "urgent": true,
        "suggest": "Reply <script>",
        "quote": "I have 3:40 or 4:20 open on Tuesday. Could you let me know by noon tomorrow so I can book the room?",
        "draft": {
          "from": "carina",
          "text": "Hi Ms. Okafor, 4:20 on Tuesday works for both of us. See you then.",
          "sign": "Carina",
          "effect": "Tue 4:20 on the family calendar",
          "effect_done": "Tue 4:20 is on the family calendar"
        },
        "say": [
          "Send it",
          "Make it 3:40",
          "Skip"
        ]
      },
      {
        "sender": "Dana \"D\" <dana@example.com>",
        "role": "Friend · coffee on Friday",
        "for": [
          "carina"
        ],
        "summary": "Asks to move Friday coffee in Ballard from 10:00 to 10:30. Your morning is open until noon.",
        "short": "<img src=x onerror=\"document.body.innerHTML=1\">",
        "due": "by Friday",
        "suggest": "Say yes, move the event",
        "quote": "Any chance we could do 10:30 instead of 10? School drop-off is running long this week.",
        "draft": {
          "from": "carina",
          "text": "Hi Dana, 10:30 is fine. Same table by the window. See you Friday!",
          "sign": "Carina",
          "effect": "Friday coffee moves to 10:30",
          "effect_done": "Friday coffee moved to 10:30"
        },
        "say": [
          "Send it",
          "Ask for 11 instead",
          "Skip"
        ]
      },
      {
        "sender": "Sound Plumbing",
        "role": "Contractor · estimate 2281",
        "for": [
          "jacob"
        ],
        "summary": "$640 to replace the kitchen shutoff valve. Earliest slot Monday 9–11; hold lapses Wednesday.",
        "short": "$640 shutoff valve · Monday 9–11",
        "due": "hold ends Wed",
        "suggest": "Approve Monday 9–11",
        "quote": "Parts and labor come to $640. We can hold Monday 9–11 for you until end of day Wednesday.",
        "draft": {
          "from": "jacob",
          "text": "Thanks. Please go ahead at $640, Monday 9–11 works. Side gate will be unlocked.",
          "sign": "Jacob",
          "effect": "Mon 9–11 on Jacob’s calendar",
          "effect_done": "Mon 9–11 is on Jacob’s calendar"
        },
        "say": [
          "Send it",
          "Ask for an afternoon slot",
          "Skip"
        ]
      }
    ],
    "handled": {
      "count": "24",
      "breakdown": "<s>news</s>"
    },
    "fyi": [
      "<img src=x onerror=\"document.body.innerHTML=1\">",
      "</span>"
    ],
    "quote": null,
    "draft": null,
    "say": null,
    "undo_seconds": null,
    "arrived": null,
    "last": null,
    "temp": 46,
    "condition": "Light rain",
    "icon": "cloud-rain"
  },
  "data": {
    "inbox": {
      "mode": "triage",
      "focus": "1",
      "privacy": "open",
      "count": 3,
      "owners": {
        "<b>carina</b>": 2,
        "jacob": "x"
      },
      "threads": [
        {
          "sender": "<img src=x onerror=\"document.body.innerHTML=1\">",
          "role": "<u>role</u> & \"co\"",
          "for": [
            "carina",
            "<i>bob</i>"
          ],
          "summary": "</p></div><script>alert(1)</script> summary",
          "short": "Conference Tuesday: 3:40 or 4:20?",
          "due": "<b>soon</b>",
          "urgent": true,
          "suggest": "Reply <script>",
          "quote": "I have 3:40 or 4:20 open on Tuesday. Could you let me know by noon tomorrow so I can book the room?",
          "draft": {
            "from": "carina",
            "text": "Hi Ms. Okafor, 4:20 on Tuesday works for both of us. See you then.",
            "sign": "Carina",
            "effect": "Tue 4:20 on the family calendar",
            "effect_done": "Tue 4:20 is on the family calendar"
          },
          "say": [
            "Send it",
            "Make it 3:40",
            "Skip"
          ]
        },
        {
          "sender": "Dana \"D\" <dana@example.com>",
          "role": "Friend · coffee on Friday",
          "for": [
            "carina"
          ],
          "summary": "Asks to move Friday coffee in Ballard from 10:00 to 10:30. Your morning is open until noon.",
          "short": "<img src=x onerror=\"document.body.innerHTML=1\">",
          "due": "by Friday",
          "suggest": "Say yes, move the event",
          "quote": "Any chance we could do 10:30 instead of 10? School drop-off is running long this week.",
          "draft": {
            "from": "carina",
            "text": "Hi Dana, 10:30 is fine. Same table by the window. See you Friday!",
            "sign": "Carina",
            "effect": "Friday coffee moves to 10:30",
            "effect_done": "Friday coffee moved to 10:30"
          },
          "say": [
            "Send it",
            "Ask for 11 instead",
            "Skip"
          ]
        },
        {
          "sender": "Sound Plumbing",
          "role": "Contractor · estimate 2281",
          "for": [
            "jacob"
          ],
          "summary": "$640 to replace the kitchen shutoff valve. Earliest slot Monday 9–11; hold lapses Wednesday.",
          "short": "$640 shutoff valve · Monday 9–11",
          "due": "hold ends Wed",
          "suggest": "Approve Monday 9–11",
          "quote": "Parts and labor come to $640. We can hold Monday 9–11 for you until end of day Wednesday.",
          "draft": {
            "from": "jacob",
            "text": "Thanks. Please go ahead at $640, Monday 9–11 works. Side gate will be unlocked.",
            "sign": "Jacob",
            "effect": "Mon 9–11 on Jacob’s calendar",
            "effect_done": "Mon 9–11 is on Jacob’s calendar"
          },
          "say": [
            "Send it",
            "Ask for an afternoon slot",
            "Skip"
          ]
        }
      ],
      "handled": {
        "count": "24",
        "breakdown": "<s>news</s>"
      },
      "fyi": [
        "<img src=x onerror=\"document.body.innerHTML=1\">",
        "</span>"
      ],
      "quote": null,
      "draft": null,
      "say": null,
      "undo_seconds": null,
      "arrived": null,
      "last": null,
      "scene": {
        "horizon": 300,
        "scale": 0.62
      }
    },
    "weather": {
      "temp": 46,
      "condition": "Light rain",
      "icon": "cloud-rain"
    }
  },
  "args": {},
  "status": null,
  "presence": null
};
